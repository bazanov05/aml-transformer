#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint> // needed for uint64_t

class Tokenizer{
    private:
    // dict used for encoding (text -> id)
    std::unordered_map<uint64_t, int> merge_rules;

    // dict used for decoding (id -> text)
    std::unordered_map<int, std::string> vocab;
    std::size_t target_vocab_size;  // max volume of tokenizer

    std::vector<int> encode(const std::string& text);

    public:
    Tokenizer(std::size_t target_vocab_size);
    void train(const std::string& text);
    int get_vocab_size(void) const;
    bool has_merge_rule(uint64_t pair_key) const;
};
