from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Tuple, Any, List
import uuid
import json
from cryptography.fernet import Fernet
import os
import structlog
from sqlalchemy.orm import Session
from db.database import MaskingMapping

logger = structlog.get_logger()

class PrivacyMasker:
    def __init__(self, model_name: str = "microsoft/phi-3-mini"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.cipher = Fernet(Fernet.generate_key())
    
    def _generate_mask_token(self) -> str:
        """Generate a unique mask token"""
        return f"<MASK_{uuid.uuid4().hex[:8]}>"
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a value"""
        return self.cipher.encrypt(value.encode()).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a value"""
        return self.cipher.decrypt(encrypted_value.encode()).decode()

class DatabaseMasker(PrivacyMasker):
    def mask_db_value(self, value: Any) -> Tuple[str, Dict]:
        """Mask a database value"""
        if not isinstance(value, (str, int, float)):
            return str(value), {}
        
        mask_token = self._generate_mask_token()
        encrypted_value = self._encrypt_value(str(value))
        
        return mask_token, {mask_token: encrypted_value}
    
    def unmask_db_value(self, masked_value: str, mapping: Dict) -> str:
        """Unmask a database value"""
        if masked_value in mapping:
            return self._decrypt_value(mapping[masked_value])
        return masked_value
    
    def store_mapping(self, doc_id: str, mapping: Dict, db: Session):
        """Store masking mapping in database"""
        db_mapping = MaskingMapping(
            id=doc_id,
            mapping=mapping
        )
        db.add(db_mapping)
        db.commit()

class DocumentMasker(PrivacyMasker):
    def mask_document(self, content: bytes) -> Tuple[str, Dict]:
        """Mask sensitive information in a document"""
        # Convert bytes to string if needed
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Split into chunks for processing
        chunks = self._split_into_chunks(content)
        
        masked_chunks = []
        mapping = {}
        
        for chunk in chunks:
            masked_chunk, chunk_mapping = self._mask_chunk(chunk)
            masked_chunks.append(masked_chunk)
            mapping.update(chunk_mapping)
        
        return "\n".join(masked_chunks), mapping
    
    def unmask_document(self, masked_content: str, mapping: Dict) -> str:
        """Unmask a document using stored mapping"""
        unmasked_content = masked_content
        for mask_token, encrypted_value in mapping.items():
            if mask_token in unmasked_content:
                decrypted_value = self._decrypt_value(encrypted_value)
                unmasked_content = unmasked_content.replace(mask_token, decrypted_value)
        return unmasked_content
    
    def _split_into_chunks(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Split content into manageable chunks"""
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _mask_chunk(self, chunk: str) -> Tuple[str, Dict]:
        """Mask sensitive information in a text chunk"""
        # Tokenize
        inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.sigmoid(outputs.logits)
        
        # Process predictions and create mapping
        masked_chunk = chunk
        mapping = {}
        
        # Apply masking based on model predictions
        # This is a simplified version - you might want to add more sophisticated logic
        for i, pred in enumerate(predictions[0]):
            if pred > 0.5:  # Threshold for sensitive information
                token = self.tokenizer.decode([inputs.input_ids[0][i]])
                if len(token.strip()) > 0:
                    mask_token = self._generate_mask_token()
                    encrypted_value = self._encrypt_value(token)
                    mapping[mask_token] = encrypted_value
                    masked_chunk = masked_chunk.replace(token, mask_token)
        
        return masked_chunk, mapping
