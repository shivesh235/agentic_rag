import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import logging
from typing import Tuple
import huggingface_hub

class ModelManager:
    _instance = None
    _model = None
    _tokenizer = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "microsoft/phi-3-mini")
        self.cache_dir = Path.home() / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self._initialize()
    
    def _initialize(self):
        """Initialize the model and tokenizer"""
        try:
            self.logger.info(f"Initializing model: {self.model_name}")
            
            # Ensure model is downloaded
            if not self._is_model_cached():
                self.logger.info(f"Downloading {self.model_name}")
                huggingface_hub.snapshot_download(
                    self.model_name,
                    cache_dir=str(self.cache_dir)
                )
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # Load model with lower precision for memory efficiency
            self.logger.info("Loading model...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16,  # Use half precision
                trust_remote_code=True
            )
            
            self.logger.info("Model initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def _is_model_cached(self) -> bool:
        """Check if model is already cached"""
        model_path = self.cache_dir / self.model_name
        return model_path.exists()
    
    def get_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Get the model and tokenizer instances"""
        if self._model is None or self._tokenizer is None:
            self._initialize()
        return self._model, self._tokenizer
    
    def generate_text(self, prompt: str, max_length: int = 200) -> str:
        """Generate text using the model"""
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7
                )
            
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            raise
