#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint> // needed for uint64_t
#include <nlohmann/json.hpp>    // for json format
#include <fstream>  // to work with files

using json = nlohmann::json;

class Tokenizer{
    private:
    // dict used for encoding (text -> id)
    std::unordered_map<uint64_t, int> merge_rules;

    // dict used for decoding (id -> text)
    std::unordered_map<int, std::string> vocab;
    std::size_t target_vocab_size;  // max volume of tokenizer

    std::vector<uint64_t> find_all_existing_pairs(const std::vector<int>& tokens);
    uint64_t find_pair_with_min_id(const std::vector<uint64_t>& all_pairs);

    public:
    Tokenizer(std::size_t target_vocab_size);
    void train(const std::string& text);
    int get_vocab_size(void) const;
    bool has_merge_rule(uint64_t pair_key) const;
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids) const;
    void save(const std::string& filepath);
    void load(const std::string& filepath);
};
