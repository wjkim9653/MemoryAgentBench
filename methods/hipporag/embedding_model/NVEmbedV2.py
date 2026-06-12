from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)


class NVEmbedV2EmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        self._patch_transformers_compat()
        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
        embedding_device = getattr(self.global_config, "embedding_device", None)
        if embedding_device and self.embedding_config.model_init_params.get("device_map") is None:
            self.embedding_model = self.embedding_model.to(embedding_device)
        self.embedding_model.eval()
        self.embedding_dim = self.embedding_model.config.hidden_size

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.
        
        Returns:
            None
        """

        model_init_params = {
            "pretrained_model_name_or_path": self.embedding_model_name,
            "trust_remote_code": True,
        }
        embedding_torch_dtype = getattr(self.global_config, "embedding_torch_dtype", "auto")
        if embedding_torch_dtype:
            model_init_params["torch_dtype"] = embedding_torch_dtype

        embedding_device_map = getattr(self.global_config, "embedding_device_map", None)
        if embedding_device_map:
            model_init_params["device_map"] = embedding_device_map

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": model_init_params,
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def _patch_transformers_compat(self) -> None:
        self._patch_tied_weights_keys_compat()
        self._patch_dynamic_cache_compat()

    def _patch_tied_weights_keys_compat(self) -> None:
        try:
            from transformers.modeling_utils import PreTrainedModel
        except Exception as exc:
            logger.warning(f"Could not patch transformers tied-weight compatibility: {exc}")
            return

        if hasattr(PreTrainedModel, "all_tied_weights_keys"):
            return

        def normalize_tied_weights_keys(tied_weights_keys):
            if tied_weights_keys is None:
                return {}
            if isinstance(tied_weights_keys, dict):
                return tied_weights_keys
            return {key: None for key in tied_weights_keys}

        def get_all_tied_weights_keys(model):
            if "_all_tied_weights_keys_compat" in model.__dict__:
                return normalize_tied_weights_keys(model.__dict__["_all_tied_weights_keys_compat"])
            return normalize_tied_weights_keys(getattr(model, "_tied_weights_keys", None))

        def set_all_tied_weights_keys(model, tied_weights_keys):
            model.__dict__["_all_tied_weights_keys_compat"] = normalize_tied_weights_keys(tied_weights_keys)

        PreTrainedModel.all_tied_weights_keys = property(
            get_all_tied_weights_keys,
            set_all_tied_weights_keys,
        )

    def _patch_dynamic_cache_compat(self) -> None:
        try:
            from transformers.cache_utils import DynamicCache
        except Exception as exc:
            logger.warning(f"Could not patch transformers DynamicCache compatibility: {exc}")
            return

        if not hasattr(DynamicCache, "from_legacy_cache"):

            @classmethod
            def from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                if past_key_values is None:
                    return cache

                for layer_idx, layer_past in enumerate(past_key_values):
                    if layer_past is None:
                        continue
                    key_states, value_states = layer_past[:2]
                    try:
                        cache.update(key_states, value_states, layer_idx)
                    except TypeError:
                        cache.update(key_states, value_states, layer_idx, cache_kwargs=None)
                return cache

            DynamicCache.from_legacy_cache = from_legacy_cache

        if not hasattr(DynamicCache, "to_legacy_cache"):

            def to_legacy_cache(cache):
                key_cache = getattr(cache, "key_cache", None)
                value_cache = getattr(cache, "value_cache", None)
                if key_cache is None or value_cache is None:
                    return tuple()
                return tuple(zip(key_cache, value_cache))

            DynamicCache.to_legacy_cache = to_legacy_cache

    # def _add_eos(self, texts: List[str]) -> List[str]:
    #     # Adds EOS token to each text
    #     return [text + self.embedding_model.tokenizer.eos_token for text in texts]

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
            # del params["instruction"]

        batch_size = params.pop("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            params["prompts"] = texts  # self._add_eos(texts=texts)
            results = self.embedding_model.encode(**params)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                params["prompts"] = texts[i:i + batch_size]
                results.append(self.embedding_model.encode(**params))
                pbar.update(batch_size)
            pbar.close()
            results = torch.cat(results, dim=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
