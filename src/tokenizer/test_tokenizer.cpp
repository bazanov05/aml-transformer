#include <gtest/gtest.h>
#include "aml_tokenizer.hpp"

// helper to construct a 64-bit pair key for assertions
uint64_t make_pair_key(int left, int right) {
    return (static_cast<uint64_t>(left) << 32) | static_cast<uint32_t>(right);
}

// 1. Standard Happy Path
TEST(TokenizerTest, TrainMethodBasic) {
    Tokenizer tokenizer(260); 
    std::string text = "ab ab"; // 'a' = 97, 'b' = 98
    
    tokenizer.train(text);
    
    // Total vocabulary should grow by 1 (256 base bytes + 1 new token)
    EXPECT_EQ(tokenizer.get_vocab_size(), 257);
    
    // Verify that the pair (97, 98) was successfully registered
    uint64_t expected_pair = make_pair_key(97, 98);
    EXPECT_TRUE(tokenizer.has_merge_rule(expected_pair));
}

// 2. Edge Case: Empty Input String
TEST(TokenizerTest, TrainEmptyString) {
    Tokenizer tokenizer(300);
    std::string text = "";
    
    EXPECT_NO_THROW(tokenizer.train(text));
    // Size should remain exactly 256 since no tokens could be formed
    EXPECT_EQ(tokenizer.get_vocab_size(), 256);
}

// 3. Edge Case: Single Character Input (No adjacent pairs exist)
TEST(TokenizerTest, TrainSingleCharacter) {
    Tokenizer tokenizer(300);
    std::string text = "a";
    
    EXPECT_NO_THROW(tokenizer.train(text));
    EXPECT_EQ(tokenizer.get_vocab_size(), 256);
}

// 4. Structural Case: Hierarchical Multi-Iteration Merges
TEST(TokenizerTest, TrainHierarchicalPattern) {
    Tokenizer tokenizer(258); // Fits exactly 2 custom tokens
    std::string text = "abababab"; 
    
    tokenizer.train(text);
    
    // Iteration 1: 'a' + 'b' (97, 98) -> 256
    // Iteration 2: 256 + 256 -> 257
    EXPECT_EQ(tokenizer.get_vocab_size(), 258);
    EXPECT_TRUE(tokenizer.has_merge_rule(make_pair_key(97, 98)));
    EXPECT_TRUE(tokenizer.has_merge_rule(make_pair_key(256, 256)));
}

// 5. Hard Edge Case: No Frequent Pairs (Every pair occurs exactly once)
TEST(TokenizerTest, TrainNoFrequentPairs) {
    Tokenizer tokenizer(300);
    std::string text = "abcdefg"; // All adjacent pairs appear exactly once
    
    tokenizer.train(text);
    
    // Should break early and refuse to merge pairs with a frequency of 1
    EXPECT_EQ(tokenizer.get_vocab_size(), 256);
}

// 6. Hard Edge Case: Non-ASCII UTF-8 Data (Verifies  unsigned char casting works)
TEST(TokenizerTest, TrainUtf8SpecialCharacters) {
    Tokenizer tokenizer(260);
    // "ł" is a multi-byte character in UTF-8, using bytes > 127
    std::string text = "łó łó łó"; 
    
    EXPECT_NO_THROW(tokenizer.train(text));
    
    // If sign extension was broken, this would crash or fail to merge
    EXPECT_GT(tokenizer.get_vocab_size(), 256);
}

// 7. Limit Case: Target Vocabulary Size is Smaller Than Starting Base Size
TEST(TokenizerTest, TrainTargetSizeSmallerThanBase) {
    Tokenizer tokenizer(100); // 100 < 256 base bytes
    std::string text = "ab ab ab";
    
    // The while condition (vocab.size() < target_vocab_size) should immediately evaluate to false
    EXPECT_NO_THROW(tokenizer.train(text));
    EXPECT_EQ(tokenizer.get_vocab_size(), 256);
}

// 1. Happy Path - no pairs
TEST(TokenizerTest, EncodeMethodBasic) {
    Tokenizer tokenizer(500); 
    std::string text_to_train = "abcdef";
    
    tokenizer.train(text_to_train);

    std::string text_to_encode = "abc";
    EXPECT_EQ(tokenizer.encode(text_to_encode), std::vector<int>({97, 98, 99}));
}

// 2. One char as text to encode
TEST(TokenizerTest, EncodeOneChar) {
    Tokenizer tokenizer(500); 
    std::string text_to_train = "abcdef";
    
    tokenizer.train(text_to_train);

    std::string text_to_encode = "a";
    
    EXPECT_EQ(tokenizer.encode(text_to_encode).size(), 1);
    EXPECT_EQ(tokenizer.encode(text_to_encode), std::vector<int>({97}));
}

// 3. Empty string as text to encode
TEST(TokenizerTest, EncodeEmtyString) {
    Tokenizer tokenizer(500); 
    std::string text_to_train = "abcdef";
    
    tokenizer.train(text_to_train);

    std::string text_to_encode = "";
    
    EXPECT_NO_THROW(tokenizer.encode(text_to_encode));
    EXPECT_EQ(tokenizer.encode(text_to_encode).size(), 0);
}

// 4. Unseen sequence as text to encode
TEST(TokenizerTest, EncodeUnseenSequence) {
    Tokenizer tokenizer(500); 
    std::string text_to_train = "abcdef";
    
    tokenizer.train(text_to_train);

    std::string text_to_encode = "xyz";
    
    EXPECT_NO_THROW(tokenizer.encode(text_to_encode));
    EXPECT_EQ(tokenizer.encode(text_to_encode).size(), 3);
}