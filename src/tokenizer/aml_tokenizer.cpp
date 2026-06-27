#include "aml_tokenizer.hpp"


Tokenizer::Tokenizer(std::size_t target_vocab_size) : target_vocab_size(target_vocab_size){
    // fill the vocabulary map with the 256 base byte strings
    for (int i = 0; i < 256; ++i) {
        vocab[i] = std::string(1, static_cast<unsigned char>(i));
    }
}