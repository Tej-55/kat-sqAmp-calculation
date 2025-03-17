import re
from itertools import cycle
from collections import Counter, OrderedDict
import random
from torchtext.vocab import vocab # type: ignore

# Special token indices
BOS_IDX = 0  # Beginning of Sequence
PAD_IDX = 1  # Padding
EOS_IDX = 2  # End of Sequence
UNK_IDX = 3  # Unknown Token
SEP_IDX = 4  # Separator Token

# Special token symbols
SPECIAL_SYMBOLS = ['<START>', '<PAD>', '<END>', '<UNK>', '<SEP>']

class Tokenizer:
    """
    Tokenizer for processing symbolic mathematical expressions.
    """
    def __init__(self, df, index_token_pool_size, momentum_token_pool_size, special_symbols, UNK_IDX, to_replace=False):
        self.amps = df.amplitude.tolist()
        self.sqamps = df.squared_amplitude.tolist()
        
        # Generate token pools
        self.tokens_pool = [f"INDEX_{i}" for i in range(index_token_pool_size)]
        self.momentum_pool = [f"MOMENTUM_{i}" for i in range(momentum_token_pool_size)]
        
        # Regular expression patterns for token replacement
        self.pattern_momentum = re.compile(r'\b[ijkl]_\d{1,}\b')
        self.pattern_num_123 = re.compile(r'\b(?![ps]_)\w+_\d{1,}\b')
        self.pattern_special = re.compile(r'\b\w+_+\w+\b\\')
        self.pattern_underscore_curly = re.compile(r'\b\w+_{')
        self.pattern_prop = re.compile(r'Prop')
        self.pattern_int = re.compile(r'int\{')
        self.pattern_operators = {
            '+': re.compile(r'\+'), '-': re.compile(r'-'), '*': re.compile(r'\*'),
            ',': re.compile(r','), '^': re.compile(r'\^'), '%': re.compile(r'%'),
            '}': re.compile(r'\}'), '(': re.compile(r'\('), ')': re.compile(r'\)')
        }
        self.pattern_mass = re.compile(r'\b\w+_\w\b')
        self.pattern_s = re.compile(r'\b\w+_\d{2,}\b')
        self.pattern_reg_prop = re.compile(r'\b\w+_\d{1}\b')
        self.pattern_antipart = re.compile(r'(\w)_\w+_\d+\(X\)\^\(\*\)')
        self.pattern_part = re.compile(r'(\w)_\w+_\d+\(X\)')
        self.pattern_index = re.compile(r'\b\w+_\w+_\d{2,}\b')
        
        self.special_symbols = special_symbols
        self.UNK_IDX = UNK_IDX
        self.to_replace = to_replace
        
        self.src_itos = {}
        self.tgt_itos = {}

    @staticmethod
    def remove_whitespace(expression):
        """Remove all forms of whitespace from the expression."""
        return re.sub(r'\s+', '', expression)

    @staticmethod
    def split_expression(expression):
        """Split the expression by space delimiter."""
        return re.split(r' ', expression)

    def build_tgt_vocab(self):
        """Build vocabulary for target sequences."""
        counter = Counter()
        for eqn in self.sqamps:
            counter.update(self.tgt_tokenize(eqn))
        voc = vocab(OrderedDict(counter), specials=self.special_symbols[:], special_first=True)
        voc.set_default_index(self.UNK_IDX)
        return voc

    def build_src_vocab(self, seed=42):
        """Build vocabulary for source sequences."""
        counter = Counter()
        for diag in self.amps:
            counter.update(self.src_tokenize(diag, seed))
        voc = vocab(OrderedDict(counter), specials=self.special_symbols[:], special_first=True)
        voc.set_default_index(self.UNK_IDX)
        return voc
    
    def src_replace(self, ampl, seed=42):
        """Replace indexed and momentum variables with tokenized equivalents."""
        ampl = self.remove_whitespace(ampl)
        
        random.seed(seed)
        token_cycle = cycle(random.sample(self.tokens_pool, len(self.tokens_pool)))
        momentum_cycle = cycle(random.sample(self.momentum_pool, len(self.momentum_pool)))
        
        # Replace momentum tokens
        temp_ampl = ampl
        momentum_mapping = {match: next(momentum_cycle) for match in set(self.pattern_momentum.findall(ampl))}
        for key, value in momentum_mapping.items():
            temp_ampl = temp_ampl.replace(key, value)
        
        # Replace index tokens
        num_123_mapping = {match: next(token_cycle) for match in set(self.pattern_num_123.findall(ampl))}
        for key, value in num_123_mapping.items():
            temp_ampl = temp_ampl.replace(key, value)

        # Replace pattern index tokens
        pattern_index_mapping = {match: f"{'_'.join(match.split('_')[:-1])} {next(token_cycle)}"
                for match in set(self.pattern_index.findall(ampl))
            }
        for key, value in pattern_index_mapping.items():
            temp_ampl = temp_ampl.replace(key, value)
            
        return temp_ampl
    
    def src_tokenize(self, ampl, seed=42):
        """Tokenize source expression, optionally applying replacements."""
        temp_ampl = self.src_replace(ampl, seed) if self.to_replace else ampl
        temp_ampl = temp_ampl.replace('\\\\', '\\').replace('\\', ' \\ ').replace('%', '')

        temp_ampl = self.pattern_underscore_curly.sub(lambda match: f' {match.group(0)} ', temp_ampl)

        
        for symbol, pattern in self.pattern_operators.items():
            temp_ampl = pattern.sub(f' {symbol} ', temp_ampl)
        
        temp_ampl = re.sub(r' {2,}', ' ', temp_ampl)
        return [token for token in self.split_expression(temp_ampl) if token]

    def tgt_tokenize(self, sqampl):
        """Tokenize target expression."""
        sqampl = self.remove_whitespace(sqampl)
        temp_sqampl = sqampl
        
        for symbol, pattern in self.pattern_operators.items():
            temp_sqampl = pattern.sub(f' {symbol} ', temp_sqampl)
        
        for pattern in [self.pattern_reg_prop, self.pattern_mass, self.pattern_s]:
            temp_sqampl = pattern.sub(lambda match: f' {match.group(0)} ', temp_sqampl)
        
        temp_sqampl = re.sub(r' {2,}', ' ', temp_sqampl)
        return [token for token in self.split_expression(temp_sqampl) if token]
    
    def src_decode(self, token_ids, skip_special_tokens=True):
        """Convert source token IDs back to text"""
        tokens = [self.src_itos.get(id, '') for id in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_symbols]
        return ' '.join(tokens)

    def tgt_decode(self, token_ids, skip_special_tokens=True):
        """Convert target token IDs back to text"""
        tokens = [self.tgt_itos.get(id, '') for id in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_symbols]
        return ' '.join(tokens)
    
def create_tokenizer(df, index_pool_size=100, momentum_pool_size=100):
    """Create a tokenizer and build source and target vocabularies."""
    
    tokenizer = Tokenizer(df, index_pool_size, momentum_pool_size, SPECIAL_SYMBOLS, UNK_IDX, to_replace=True)
    
    src_vocab = tokenizer.build_src_vocab()
    src_itos = {value: key for key, value in src_vocab.get_stoi().items()}
    tgt_vocab = tokenizer.build_tgt_vocab()
    tgt_itos = {value: key for key, value in tgt_vocab.get_stoi().items()}
    
    tokenizer.src_itos = src_itos
    tokenizer.tgt_itos = tgt_itos

    return tokenizer, src_vocab, tgt_vocab, src_itos, tgt_itos
    
def normalize_indices(tokenizer, expressions, index_token_pool_size=50, momentum_token_pool_size=50):
    # Function to replace indices with a new set of tokens for each expression
    def replace_indices(token_list, index_map):
        new_index = (f"INDEX_{i}" for i in range(index_token_pool_size))  # Local generator for new indices
        new_tokens = []
        for token in token_list:
            if "INDEX_" in token:
                if token not in index_map:
                    try:
                        index_map[token] = next(new_index)
                    except StopIteration:
                        # Handle the case where no more indices are available
                        raise ValueError("Ran out of unique indices, increase token_pool_size")
                new_tokens.append(index_map[token])
            else:
                new_tokens.append(token)
        return new_tokens

    def replace_momenta(token_list, index_map):
        new_index = (f"MOMENTUM_{i}" for i in range(momentum_token_pool_size))  # Local generator for new indices
        new_tokens = []
        for token in token_list:
            if "MOMENTUM_" in token:
                if token not in index_map:
                    try:
                        index_map[token] = next(new_index)
                    except StopIteration:
                        # Handle the case where no more indices are available
                        raise ValueError("Ran out of unique indices, increase momentum_token_pool_size")
                new_tokens.append(index_map[token])
            else:
                new_tokens.append(token)
        return new_tokens

    normalized_expressions = []
    # Replace indices in each expression randomly
    for expr in expressions:
        toks = tokenizer.src_tokenize(expr)
        print(toks)
        normalized_expressions.append(replace_momenta(replace_indices(toks, {}), {}))

    return normalized_expressions