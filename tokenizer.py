"""
Code-aware tokenizer for CodeLACE.
"""

import re
import torch
from typing import List, Dict, Optional


class CodeTokenizer:
    """Code-aware tokenizer for programming languages."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = self._build_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.cls_token = '<CLS>'
        self.sep_token = '<SEP>'

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.cls_token_id = self.token_to_id[self.cls_token]
        self.sep_token_id = self.token_to_id[self.sep_token]

    def _build_vocab(self) -> List[str]:
        """Build vocabulary for code tokenization."""
        # Special tokens
        special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>']

        # Programming keywords
        keywords = [
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'import', 'from', 'return', 'yield', 'break', 'continue', 'pass',
            'public', 'private', 'protected', 'static', 'final', 'abstract',
            'function', 'var', 'let', 'const', 'int', 'float', 'string', 'bool',
            'true', 'false', 'null', 'undefined', 'void', 'new', 'this', 'super'
        ]

        # Operators and punctuation
        operators = [
            '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
            '&&', '||', '!', '&', '|', '^', '<<', '>>', '++', '--',
            '(', ')', '[', ']', '{', '}', ';', ':', ',', '.', '?'
        ]

        # Common identifiers and literals
        common_tokens = [
            'i', 'j', 'k', 'x', 'y', 'z', 'n', 'm', 'len', 'size', 'count',
            'data', 'result', 'value', 'item', 'key', 'index', 'temp',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'main', 'init', 'get', 'set', 'add', 'remove', 'update', 'delete'
        ]

        # Build complete vocabulary
        vocab = special_tokens + keywords + operators + common_tokens

        # Add more generic tokens to reach vocab_size
        for i in range(len(vocab), self.vocab_size):
            vocab.append(f'<TOKEN_{i}>')

        return vocab[:self.vocab_size]

    def tokenize(self, code: str, language: str = 'python') -> List[str]:
        """Tokenize source code into tokens."""
        # Basic tokenization using regex
        # This is a simplified tokenizer - in practice, you'd use a proper parser

        # Remove comments
        if language == 'python':
            code = re.sub(r'#.*', '', code)
        elif language in ['java', 'javascript', 'cpp']:
            code = re.sub(r'//.*', '', code)
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Tokenize using regex
        pattern = r'\b\w+\b|[^\w\s]'
        tokens = re.findall(pattern, code.lower())

        # Filter out empty tokens and whitespace
        tokens = [token for token in tokens if token.strip()]

        return tokens

    def encode(self, code: str, language: str = 'python', max_length: int = 512) -> torch.Tensor:
        """Encode source code to token IDs."""
        tokens = self.tokenize(code, language)

        # Add special tokens
        tokens = [self.cls_token] + tokens + [self.sep_token]

        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.unk_token_id)

        # Truncate or pad to max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to source code."""
        tokens = []
        for token_id in token_ids:
            if token_id.item() in self.id_to_token:
                token = self.id_to_token[token_id.item()]
                if token not in [self.pad_token, self.cls_token, self.sep_token]:
                    tokens.append(token)

        return ' '.join(tokens)

    def create_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask for padded sequences."""
        return (token_ids != self.pad_token_id).long()


# Test the tokenizer
if __name__ == "__main__":
    tokenizer = CodeTokenizer()

    # Test code
    code = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """

    print("Original code:")
    print(code)

    tokens = tokenizer.tokenize(code, 'python')
    print(f"\nTokens: {tokens}")

    encoded = tokenizer.encode(code, 'python')
    print(f"\nEncoded shape: {encoded.shape}")
    print(f"Encoded: {encoded[:20]}...")  # Show first 20 tokens

    decoded = tokenizer.decode(encoded)
    print(f"\nDecoded: {decoded}")