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

// 5. not ASCII symbols
TEST(TokenizerTest, EncodeNotASCII) {
    Tokenizer tokenizer(256); 
    std::string text_to_train = "abcdef";
    
    tokenizer.train(text_to_train);

    std::string text_to_encode = "łó";
    
    EXPECT_NO_THROW(tokenizer.encode(text_to_encode));
    EXPECT_EQ(tokenizer.encode(text_to_encode).size(), 4);
    EXPECT_EQ(tokenizer.encode(text_to_encode), std::vector<int>({197, 130, 195, 179}));
}

// 6. Rule-Rank Priority
TEST(TokenizerTest, EncodeRankPriority) {
    Tokenizer tokenizer(500); 
    std::string text_to_train = "abbcbcbcabbc"; // bc has higher priority
    
    tokenizer.train(text_to_train);

    std::string text_to_encode = "abc";
    
    EXPECT_NO_THROW(tokenizer.encode(text_to_encode));
    EXPECT_EQ(tokenizer.encode(text_to_encode).size(), 2);
    EXPECT_EQ(tokenizer.encode(text_to_encode), std::vector<int>({97, 256}));
}

// 7. Hierarchical Structural Merging
TEST(TokenizerTest, EncodeMerging) {
    Tokenizer tokenizer(500); 
    std::string text_to_train = "ababababababab";
    
    tokenizer.train(text_to_train);

    std::string text_to_encode = "abab";
    
    EXPECT_NO_THROW(tokenizer.encode(text_to_encode));
    EXPECT_EQ(tokenizer.encode(text_to_encode).size(), 1);
    EXPECT_EQ(tokenizer.encode(text_to_encode), std::vector<int>({257}));
}

// 8. Trailing Unmerged Token
TEST(TokenizerTest, EncodeTrailingUnmergedToken) {
    Tokenizer tokenizer(500); 
    std::string text_to_train = "ababababababab";
    
    tokenizer.train(text_to_train);

    std::string text_to_encode = "abc";
    
    EXPECT_NO_THROW(tokenizer.encode(text_to_encode));
    EXPECT_EQ(tokenizer.encode(text_to_encode).size(), 2);
    EXPECT_EQ(tokenizer.encode(text_to_encode), std::vector<int>({256, 99}));
}

// 1. Base Case: Standard ASCII string without any merged pairs
TEST(TokenizerTest, DecodeMethodBasic) {
    Tokenizer tokenizer(500); 
    // Assuming tokenizer has base vocab 0-255 initialized
    std::vector<int> input_ids = {97, 98, 99}; // 'a', 'b', 'c'
    
    EXPECT_EQ(tokenizer.decode(input_ids), "abc");
}

// 2. Base Case: Reconstructing merged tokens
TEST(TokenizerTest, DecodeMergedTokens) {
    Tokenizer tokenizer(500); 
    tokenizer.train("abab"); // 'ab' becomes 256
    
    std::vector<int> input_ids = {256, 256};
    
    EXPECT_EQ(tokenizer.decode(input_ids), "abab");
}

// 3. Edge Case: Empty vector input
TEST(TokenizerTest, DecodeEmptyVector) {
    Tokenizer tokenizer(500); 
    std::vector<int> input_ids = {};
    
    EXPECT_EQ(tokenizer.decode(input_ids), "");
}

// 4. Edge Case: Single token ID
TEST(TokenizerTest, DecodeSingleToken) {
    Tokenizer tokenizer(500); 
    std::vector<int> input_ids = {122}; // 'z'
    
    EXPECT_EQ(tokenizer.decode(input_ids), "z");
}

// 5. Edge Case: Non-ASCII UTF-8 reconstruction
TEST(TokenizerTest, DecodeNotASCII) {
    Tokenizer tokenizer(500); 
    // "łó" splits into 4 bytes: 197, 130, 195, 179
    std::vector<int> input_ids = {197, 130, 195, 179};
    
    EXPECT_EQ(tokenizer.decode(input_ids), "łó");
}

// 6. Structural Case: Mixed sequence of base bytes and merged tokens
TEST(TokenizerTest, DecodeMixedSequence) {
    Tokenizer tokenizer(500);
    tokenizer.train("abbcbcbcabbc"); // 'bc' -> 256
    
    // 'a' (97), 'bc' (256), 'bc' (256), 'z' (122)
    std::vector<int> input_ids = {97, 256, 256, 122};
    
    EXPECT_EQ(tokenizer.decode(input_ids), "abcbcz");
}

// 7. Security/Error Case: Out of bounds token ID (Tests the .at() method)
TEST(TokenizerTest, DecodeInvalidTokenID) {
    Tokenizer tokenizer(500); 
    std::vector<int> input_ids = {97, 9999, 98}; // 9999 does not exist
    
    EXPECT_THROW(tokenizer.decode(input_ids), std::out_of_range);
}

// 8. Pipeline Case: Full cycle (Train -> Encode -> Decode)
TEST(TokenizerTest, FullPipelineReconstruction) {
    Tokenizer tokenizer(500);
    std::string original_text = "Data engineering requires reliable systems.";
    
    tokenizer.train(original_text);
    std::vector<int> encoded_ids = tokenizer.encode(original_text);
    std::string decoded_text = tokenizer.decode(encoded_ids);
    
    EXPECT_EQ(decoded_text, original_text);
}