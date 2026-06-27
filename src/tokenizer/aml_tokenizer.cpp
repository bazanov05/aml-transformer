#include "aml_tokenizer.hpp"


Tokenizer::Tokenizer(std::size_t target_vocab_size) : target_vocab_size(target_vocab_size){
    // fill the vocabulary map with the 256 base byte strings
    for (int i = 0; i < 256; ++i) {
        vocab[i] = std::string(1, static_cast<unsigned char>(i));
    }
}

void Tokenizer::train(const std::string& text){
    std::vector<int> ids;   // vector of ids - each token recieves its own id
    ids.reserve(text.size());

    for(const char& symbol : text){
        // fulfill ids vector with pure base byte strings firstly 
        ids.push_back(static_cast<int>(static_cast<unsigned char>(symbol)));
    }

    // compress pairs until the vocab size is not reached 
    while(vocab.size() < this -> target_vocab_size){
        std::unordered_map<uint64_t, int> freq;   // to count the frequency of each pair 
        for(std::size_t i = 0; i < ids.size(); ++i){
            if(i + 1 < ids.size()){
                int left = ids[i];
                int right = ids[i + 1];

                // use bitwise shift 
                uint64_t pair = (static_cast<uint64_t>(left) << 32) | static_cast<uint32_t>(right);            
                freq[pair]++;
            }
        }

        uint64_t best_pair;
        int best_freq = 0;

        // find best pair based on highest freqeuncy 
        for(const auto& pair : freq){
            if(pair.second > best_freq){
                best_freq = pair.second;
                best_pair = pair.first;
            }
        }

        // if all pairs occur once - stop compressing depsite not reaching the goal vocab size 
        if(best_freq < 2){
            break;
        }
        
        // firstly vocab size is 256(ids 0-255), so new token will have id = 256
        int new_id = this -> vocab.size();
        this -> merge_rules[best_pair] = new_id;

        int right = static_cast<int>(best_pair & 0xFFFFFFFF);
        int left = static_cast<int>(best_pair >> 32);
        this -> vocab[new_id] = this -> vocab[left] + this -> vocab[right];
        
        std::vector<int> new_ids;
        new_ids.reserve(ids.size());

        // replace old tokens with newly created one for best pair
        for(std::size_t i = 0; i < ids.size();){
            if(i + 1 < ids.size() && ids[i] == left && ids[i + 1] == right){
                new_ids.push_back(new_id);
                i += 2;
            }
            else{
                new_ids.push_back(ids[i]);
                i++;
            }
        }

        // update ids vector which now contains best pair from this iteration
        ids = std::move(new_ids);
        
    }
}

int Tokenizer::get_vocab_size() const{
    return this -> vocab.size();
}

bool Tokenizer::has_merge_rule(uint64_t pair_key) const{
    return this -> merge_rules.find(pair_key) != this -> merge_rules.end();
}