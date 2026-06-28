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

std::vector<int> Tokenizer::encode(const std::string& text){
    std::vector<int> tokens;
    tokens.reserve(text.size());

    // firstly split input text into single bytes
    for(const auto& symbol : text){
        tokens.push_back(static_cast<int>(static_cast<unsigned char>(symbol)));
    }

    bool pair_was_made = true;  // track if the pair was made - if not, out text is fully splitted

    while(pair_was_made){
        std::vector<int> new_tokens;
        new_tokens.reserve(tokens.size());
        
        // find all possible pair of ids in current state
        std::vector<uint64_t> all_pairs = this -> find_all_existing_pairs(tokens);

        // if no pair was found - we have splitted text successfully, stop the loop
        if(all_pairs.empty()){
            pair_was_made = false;
            break;
        }

        // find the pair with minimal id
        // minimal id means that this pair occured more often during the training
        // that is why it is more important for the context 
        uint64_t min_pair = this -> find_pair_with_min_id(all_pairs);

        for(std::size_t i = 0; i < tokens.size();){
            if(i + 1 < tokens.size()){
                uint64_t left = static_cast<uint64_t>(tokens[i]);
                uint32_t right = static_cast<uint32_t>(tokens[i + 1]);
                uint64_t pair = ((left << 32) | right);
                
                // if current pair == min_pair - replace 2 tokens with this 1 pair
                if(pair == min_pair){
                    new_tokens.push_back(this -> merge_rules[pair]);
                    i += 2;
                }
                // otherwise save the left
                else{
                    new_tokens.push_back(tokens[i]);
                    i++;
                }
            }
            else{
                new_tokens.push_back(tokens[i]);
                i++;
            }
        }
        // update tokens - now it contains new created pair 
        tokens = std::move(new_tokens);
    }
    return tokens;
}

std::vector<uint64_t> Tokenizer::find_all_existing_pairs(const std::vector<int>& tokens){
    std::vector<uint64_t> all_pairs;
    all_pairs.reserve(tokens.size());
    for(std::size_t i = 0; i + 1 < tokens.size(); ++i){
            uint64_t left = static_cast<uint64_t>(tokens[i]);
            uint32_t right = static_cast<uint32_t>(tokens[i + 1]);
            uint64_t pair = ((left << 32) | right);

            // if pair is in our merge rules - push it 
            if(this -> has_merge_rule(pair)){
                all_pairs.push_back(pair);
            }
    }

    return all_pairs;
}

uint64_t Tokenizer::find_pair_with_min_id(const std::vector<uint64_t>& all_pairs){
    int min_id = this -> merge_rules[all_pairs[0]];
    uint64_t min_pair = all_pairs[0];

    for(const auto& pair : all_pairs){
        if(this -> merge_rules[pair] < min_id){
            min_id = merge_rules[pair];
            min_pair = pair;
        }
    }

    return min_pair;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const{
    std::string result = "";

    for(const auto& id : ids){
        // .at() throws out_of_range if the id was not found in vocab
        result += this -> vocab.at(id);
    }

    return result;
}

void Tokenizer::save(const std::string& filepath){
    std::ofstream file(filepath);

    if(!file.is_open()){
        throw std::runtime_error("Failed to open .json file");
    }

    json j;
    json rules;

    // serialize merge_rules
    // json cannot convert uint_64t to string
    for(auto& pair : this->merge_rules){
        rules[std::to_string(pair.first)] = pair.second;
    }
    j["merge_rules"] = rules;

    // serialize vocab as arrays of integers (bytes) to avoid UTF-8 crashes
    json vocab_json;
    for(const auto& pair : this->vocab){
        json byte_array = json::array();
        for(unsigned char c : pair.second){
            byte_array.push_back(static_cast<int>(c)); // Store raw byte as int
        }
        vocab_json[std::to_string(pair.first)] = byte_array;
    }
    j["vocab"] = vocab_json;

    file << j.dump(4);
}

void Tokenizer::load(const std::string& filepath){
    std::ifstream file(filepath);

    if(!file.is_open()){
        throw std::runtime_error("Failed to open .json file");
    }

    json j;
    file >> j; 

    this->merge_rules.clear();
    this->vocab.clear();

    // reconstruct merge_rules
    for (auto& el : j["merge_rules"].items()) {
        this->merge_rules[std::stoull(el.key())] = el.value().get<int>();
    }

    // reconstruct vocab strings from integer arrays
    for (auto& el : j["vocab"].items()) {
        std::string token_str = "";
        for (int byte_val : el.value()) {
            token_str += static_cast<char>(byte_val); // cast int back to raw byte
        }
        this->vocab[std::stoi(el.key())] = token_str;
    }
}