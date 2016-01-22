/*
 * RCNNHSWordCharClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_RCNNHSWordCharClassifier_H_
#define SRC_RCNNHSWordCharClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"

#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using word and character embeddings
template<typename xpu>
class RCNNHSWordCharClassifier {
public:
  RCNNHSWordCharClassifier() {
    _dropOut = 0.5;
  }
  ~RCNNHSWordCharClassifier() {

  }

public:
  LookupTable<xpu> _words;

  LookupTable<xpu> _hwords;

  int _wordcontext;
  int _wordSize;
  int _wordDim;
  bool _b_wordEmb_finetune;
  int _wordHiddenSize;
  int _word_cnn_iSize;

  //char
  LookupTable<xpu> _chars;
  LookupTable<xpu> _hchars;
  int _charcontext;
  int _charSize;
  int _charDim;
  bool _b_charEmb_finetune;
  int _charHiddenSize;
  int _char_cnn_iSize;

  int _token_representation_size;

  int _hiddenSize;

  UniLayer<xpu> _tanh_project;
  UniLayer<xpu> _cnn_project;
  UniLayer<xpu> _olayer_linear;

  RNN<xpu> _rnn_left;
  RNN<xpu> _rnn_right;

  //char
  UniLayer<xpu> _char_cnn_project;
  RNN<xpu> _char_rnn_left;
  RNN<xpu> _char_rnn_right;
  UniLayer<xpu> _hchar_cnn_project;
  RNN<xpu> _hchar_rnn_left;
  RNN<xpu> _hchar_rnn_right;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3 min

  int _poolmanners;

public:

  inline void init(const NRMat<dtype>& wordEmb, const NRMat<dtype>& wordhEmb, const NRMat<dtype>& charEmb, const NRMat<dtype>& hcharEmb, int wordcontext,
      int charcontext, int labelSize, int wordHiddenSize, int charHiddenSize, int hiddenSize) {
    _wordcontext = wordcontext;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();
    _poolmanners = 3;

    //char
    _charcontext = charcontext;
    _charSize = charEmb.nrows();
    _charDim = charEmb.ncols();
    _charHiddenSize = charHiddenSize;
    _char_cnn_iSize = _charDim * (2 * _charcontext + 1);

    _chars.initial(charEmb);
    _hchars.initial(hcharEmb);
    _char_rnn_left.initial(_charHiddenSize, _char_cnn_iSize, true, 30);
    _char_rnn_right.initial(_charHiddenSize, _char_cnn_iSize, false, 70);
    _char_cnn_project.initial(_charHiddenSize, 2 * _charHiddenSize, true, 40, 0);
    _hchar_rnn_left.initial(_charHiddenSize, _char_cnn_iSize, true, 80);
    _hchar_rnn_right.initial(_charHiddenSize, _char_cnn_iSize, false, 90);
    _hchar_cnn_project.initial(_charHiddenSize, 2 * _charHiddenSize, true, 120, 0);

    _labelSize = labelSize;
    _hiddenSize = hiddenSize;
    _wordHiddenSize = wordHiddenSize;
    _token_representation_size = _wordDim + _poolmanners * _charHiddenSize;

    _word_cnn_iSize = _token_representation_size * (2 * _wordcontext + 1);

    _words.initial(wordEmb);

    _hwords.initial(wordhEmb);

    _rnn_left.initial(_wordHiddenSize, _word_cnn_iSize, true, 100);
    _rnn_right.initial(_wordHiddenSize, _word_cnn_iSize, false, 110);

    _cnn_project.initial(_wordHiddenSize, 2 * _wordHiddenSize, true, 20, 0);
    _tanh_project.initial(_hiddenSize, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size, true, 50, 0);
    _olayer_linear.initial(_labelSize, _hiddenSize, false, 60, 2);

    _eval.reset();

    _remove = 0;
  }

  inline void release() {
    //char    
    _chars.release();
    _char_rnn_left.release();
    _char_rnn_right.release();
    _char_cnn_project.release();
    _hchars.release();
    _hchar_rnn_left.release();
    _hchar_rnn_right.release();
    _hchar_cnn_project.release();

    _words.release();
    _hwords.release();

    _cnn_project.release();
    _tanh_project.release();
    _olayer_linear.release();
    _rnn_left.release();
    _rnn_right.release();
  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();
      if (seq_size > 2) {
        std::cout << "error" << std::endl;
      }

      //char define

      vector<vector<Tensor<xpu, 3, dtype> > > charprime, charprimeLoss, charprimeMask;
      vector<vector<Tensor<xpu, 3, dtype> > > char_input, char_inputLoss;
      vector<vector<Tensor<xpu, 3, dtype> > > char_rnn_hidden_left, char_rnn_hidden_leftLoss;
      vector<vector<Tensor<xpu, 3, dtype> > > char_rnn_hidden_right, char_rnn_hidden_rightLoss;
      vector<vector<Tensor<xpu, 3, dtype> > > char_midhidden, char_midhiddenLoss;
      vector<vector<Tensor<xpu, 3, dtype> > > char_hidden, char_hiddenLoss;
      vector<vector<vector<Tensor<xpu, 2, dtype> > > > char_pool, char_poolLoss;
      vector<vector<vector<Tensor<xpu, 3, dtype> > > > char_poolIndex;
      vector<vector<Tensor<xpu, 2, dtype> > > char_poolmerge;
      vector<vector<Tensor<xpu, 2, dtype> > > char_poolmergeLoss;

      charprime.resize(seq_size);
      charprimeLoss.resize(seq_size);
      charprimeMask.resize(seq_size);
      char_input.resize(seq_size);
      char_inputLoss.resize(seq_size);
      char_rnn_hidden_left.resize(seq_size);
      char_rnn_hidden_leftLoss.resize(seq_size);
      char_rnn_hidden_right.resize(seq_size);
      char_rnn_hidden_rightLoss.resize(seq_size);
      char_midhidden.resize(seq_size);
      char_midhiddenLoss.resize(seq_size);
      char_hidden.resize(seq_size);
      char_hiddenLoss.resize(seq_size);
      char_pool.resize(seq_size);
      char_poolLoss.resize(seq_size);
      char_poolIndex.resize(seq_size);
      char_poolmerge.resize(seq_size);
      char_poolmergeLoss.resize(seq_size);
      for (int idx = 0; idx < seq_size; idx++) {
        int word_num = example.m_features[idx].words.size();
        charprime[idx].resize(word_num);
        charprimeLoss[idx].resize(word_num);
        charprimeMask[idx].resize(word_num);
        char_input[idx].resize(word_num);
        char_inputLoss[idx].resize(word_num);
        char_rnn_hidden_left[idx].resize(word_num);
        char_rnn_hidden_leftLoss[idx].resize(word_num);
        char_rnn_hidden_right[idx].resize(word_num);
        char_rnn_hidden_rightLoss[idx].resize(word_num);
        char_midhidden[idx].resize(word_num);
        char_midhiddenLoss[idx].resize(word_num);
        char_hidden[idx].resize(word_num);
        char_hiddenLoss[idx].resize(word_num);
        char_pool[idx].resize(word_num);
        char_poolLoss[idx].resize(word_num);
        char_poolIndex[idx].resize(word_num);
        char_poolmerge[idx].resize(word_num);
        char_poolmergeLoss[idx].resize(word_num);
        for (int idy = 0; idy < word_num; idy++) {
          char_pool[idx][idy].resize(_poolmanners);
          char_poolLoss[idx][idy].resize(_poolmanners);
          char_poolIndex[idx][idy].resize(_poolmanners);
        }

      }

      //word
      Tensor<xpu, 3, dtype> wordprime[seq_size], wordprimeLoss[seq_size], wordprimeMask[seq_size];
      Tensor<xpu, 3, dtype> wordrepresent[seq_size], wordrepresentLoss[seq_size];
      Tensor<xpu, 3, dtype> input[seq_size], inputLoss[seq_size];

      Tensor<xpu, 3, dtype> rnn_hidden_left, rnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_right, rnn_hidden_rightLoss;

      Tensor<xpu, 3, dtype> midhidden, midhiddenLoss;

      Tensor<xpu, 3, dtype> hidden[seq_size], hiddenLoss[seq_size];
      vector<Tensor<xpu, 2, dtype> > pool(seq_size * _poolmanners), poolLoss(seq_size * _poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndex(seq_size * _poolmanners);

      Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, dtype> project, projectLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;

      //initialize

      for (int idx = 0; idx < seq_size; idx++) {
        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        const Feature& feature = example.m_features[idx];
        int word_num = feature.words.size();

        //char initial
        int char_cnn_iSize = _char_cnn_iSize;
        int charHiddenSize = _charHiddenSize;
        for (int idy = 0; idy < word_num; idy++) {
          int char_num = feature.chars[idy].size();
          charprime[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);
          charprimeLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);
          charprimeMask[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_one);

          char_input[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, char_cnn_iSize), d_zero);
          char_inputLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, char_cnn_iSize), d_zero);
          char_rnn_hidden_left[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
          char_rnn_hidden_leftLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
          char_rnn_hidden_right[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
          char_rnn_hidden_rightLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
          char_midhidden[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, 2 * charHiddenSize), d_zero);
          char_midhiddenLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, 2 * charHiddenSize), d_zero);
          char_hidden[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
          char_hiddenLoss[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);

          for (int idz = 0; idz < _poolmanners; idz++) {
            char_pool[idx][idy][idz] = NewTensor<xpu>(Shape2(1, charHiddenSize), d_zero);
            char_poolLoss[idx][idy][idz] = NewTensor<xpu>(Shape2(1, charHiddenSize), d_zero);
            char_poolIndex[idx][idy][idz] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
          }

          char_poolmerge[idx][idy] = NewTensor<xpu>(Shape2(1, _poolmanners * charHiddenSize), d_zero);
          char_poolmergeLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _poolmanners * charHiddenSize), d_zero);
        }

        //word initial

        int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
        int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

        wordprime[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
        wordprimeLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
        wordprimeMask[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_one);
        wordrepresent[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
        wordrepresentLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);

        input[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);
        inputLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);

        hidden[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
        hiddenLoss[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

        if (idx == seq_size - 1) {
          rnn_hidden_left = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_leftLoss = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

          rnn_hidden_right = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
          rnn_hidden_rightLoss = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

          midhidden = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);
          midhiddenLoss = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);
        }

        offset = idx * _poolmanners;
        for (int idm = 0; idm < _poolmanners; idm++) {
          pool[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
          poolLoss[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
          poolIndex[offset + idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
        }
      }

      poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), d_zero);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), d_zero);
      project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
      projectLoss = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
      output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

      //forward propagation
      //input setting, and linear setting
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        int word_num = feature.words.size();
        int char_num = -1;

        //char
        int char_curcontext = _charcontext;
        const vector<vector<int> >& chars = feature.chars;
        srand(iter * example_num + count * seq_size + idx);

        //get charEmb
        for (int idy = 0; idy < word_num; idy++) {
          char_num = feature.chars[idy].size();
          for (int idz = 0; idz < char_num; idz++) {
            if (idx == seq_size - 1) {
              _chars.GetEmb(chars[idy][idz], charprime[idx][idy][idz]);
            } else {
              _hchars.GetEmb(chars[idy][idz], charprime[idx][idy][idz]);
            }
          }
        }

        //char dropout
        for (int idy = 0; idy < word_num; idy++) {
          char_num = feature.chars[idy].size();
          for (int idz = 0; idz < char_num; idz++) {
            dropoutcol(charprimeMask[idx][idy][idz], _dropOut);
            charprime[idx][idy][idz] = charprime[idx][idy][idz] * charprimeMask[idx][idy][idz];
          }
        }

        for (int idy = 0; idy < word_num; idy++) {
          windowlized(charprime[idx][idy], char_input[idx][idy], char_curcontext);
        }

        //char RNN
        for (int idy = 0; idy < word_num; idy++) {
          if (idx == seq_size - 1) {
            _char_rnn_left.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_left[idx][idy]);
            _char_rnn_right.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_right[idx][idy]);
          } else {
            _hchar_rnn_left.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_left[idx][idy]);
            _hchar_rnn_right.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_right[idx][idy]);

          }
        }

        for (int idy = 0; idy < word_num; idy++) {
          char_num = feature.chars[idy].size();
          for (int idz = 0; idz < char_num; idz++) {
            concat(char_rnn_hidden_left[idx][idy][idz], char_rnn_hidden_right[idx][idy][idz], char_midhidden[idx][idy][idz]);
          }
        }

        //char convolution

        for (int idy = 0; idy < word_num; idy++) {
          if (idx == seq_size - 1) {
            _char_cnn_project.ComputeForwardScore(char_midhidden[idx][idy], char_hidden[idx][idy]);
          } else {
            _hchar_cnn_project.ComputeForwardScore(char_midhidden[idx][idy], char_hidden[idx][idy]);
          }
        }

        //char pooling
        for (int idy = 0; idy < word_num; idy++) {
          if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
            avgpool_forward(char_hidden[idx][idy], char_pool[idx][idy][0], char_poolIndex[idx][idy][0]);
          }
          if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
            maxpool_forward(char_hidden[idx][idy], char_pool[idx][idy][1], char_poolIndex[idx][idy][1]);
          }
          if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
            minpool_forward(char_hidden[idx][idy], char_pool[idx][idy][2], char_poolIndex[idx][idy][2]);
          }
        }

        //charConcat2word
        for (int idy = 0; idy < word_num; idy++) {
          concat(char_pool[idx][idy], char_poolmerge[idx][idy]);
        }

        //word
        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

        const vector<int>& words = feature.words;
        word_num = words.size();
        //linear features should not be dropped out

        for (int idy = 0; idy < word_num; idy++) {
          if (idx == seq_size - 1)
            _words.GetEmb(words[idy], wordprime[idx][idy]);
          else
            _hwords.GetEmb(words[idy], wordprime[idx][idy]);
        }

        //word dropout
        for (int idy = 0; idy < word_num; idy++) {
          dropoutcol(wordprimeMask[idx][idy], _dropOut);
          wordprime[idx][idy] = wordprime[idx][idy] * wordprimeMask[idx][idy];
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          //wordrepresent[idx][idy] += wordprime[idx][idy];
          //char_poolmerge[idx][idy] = 0.0;
          concat(wordprime[idx][idy], char_poolmerge[idx][idy], wordrepresent[idx][idy]);
        }

        windowlized(wordrepresent[idx], input[idx], curcontext);

        if (idx == seq_size - 1) {
          _rnn_left.ComputeForwardScore(input[idx], rnn_hidden_left);
          _rnn_right.ComputeForwardScore(input[idx], rnn_hidden_right);

          for (int idy = 0; idy < word_num; idy++) {
            concat(rnn_hidden_left[idy], rnn_hidden_right[idy], midhidden[idy]);
          }

          _cnn_project.ComputeForwardScore(midhidden, hidden[idx]);
        } else {
          for (int idy = 0; idy < word_num; idy++) {
            hidden[idx][idy] += input[idx][idy];
          }
        }

        //word pooling
        offset = idx * _poolmanners;
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(hidden[idx], pool[offset], poolIndex[offset]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(hidden[idx], pool[offset + 1], poolIndex[offset + 1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(hidden[idx], pool[offset + 2], poolIndex[offset + 2]);
        }
      }

      // sentence
      offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
      concat(pool, poolmerge, offset);
      _tanh_project.ComputeForwardScore(poolmerge, project);
      _olayer_linear.ComputeForwardScore(project, output);

      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      //sentence
      _olayer_linear.ComputeBackwardLoss(project, output, outputLoss, projectLoss);
      _tanh_project.ComputeBackwardLoss(poolmerge, project, projectLoss, poolmergeLoss);

      offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
      unconcat(poolLoss, poolmergeLoss, offset);

      for (int idx = 0; idx < seq_size; idx++) {

        //char
        const Feature& feature = example.m_features[idx];
        int window_char = 2 * _charcontext + 1;
        int char_curcontext = _charcontext;
        const vector<vector<int> >& chars = feature.chars;

        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

        const vector<int>& words = feature.words;
        int word_num = words.size();

        //word pooling
        offset = idx * _poolmanners;
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          pool_backward(poolLoss[offset], poolIndex[offset], hiddenLoss[idx]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], hiddenLoss[idx]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], hiddenLoss[idx]);
        }

        if (idx == seq_size - 1) {
          _cnn_project.ComputeBackwardLoss(midhidden, hidden[idx], hiddenLoss[idx], midhiddenLoss);

          for (int idy = 0; idy < word_num; idy++) {
            unconcat(rnn_hidden_leftLoss[idy], rnn_hidden_rightLoss[idy], midhiddenLoss[idy]);
          }

          _rnn_left.ComputeBackwardLoss(input[idx], rnn_hidden_left, rnn_hidden_leftLoss, inputLoss[idx]);
          _rnn_right.ComputeBackwardLoss(input[idx], rnn_hidden_right, rnn_hidden_rightLoss, inputLoss[idx]);
        } else {
          for (int idy = 0; idy < word_num; idy++) {
            inputLoss[idx][idy] += hiddenLoss[idx][idy];
          }
        }

        windowlized_backward(wordrepresentLoss[idx], inputLoss[idx], curcontext);

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          //wordprimeLoss[idx][idy] += wordrepresentLoss[idx][idy];
          unconcat(wordprimeLoss[idx][idy], char_poolmergeLoss[idx][idy], wordrepresentLoss[idx][idy]);
          //char_poolmergeLoss[idy] = 0.0;
        }

        //word dropout
        if (idx == seq_size - 1) {
          if (_words.bEmbFineTune()) {
            for (int idy = 0; idy < word_num; idy++) {
              wordprimeLoss[idx][idy] = wordprimeLoss[idx][idy] * wordprimeMask[idx][idy];
              _words.EmbLoss(words[idy], wordprimeLoss[idx][idy]);
            }
          }
        } else {
          if (_hwords.bEmbFineTune()) {
            for (int idy = 0; idy < word_num; idy++) {
              wordprimeLoss[idx][idy] = wordprimeLoss[idx][idy] * wordprimeMask[idx][idy];
              _hwords.EmbLoss(words[idy], wordprimeLoss[idx][idy]);
            }
          }
        }

        //char loss backward propagation
        for (int idy = 0; idy < word_num; idy++) {
          unconcat(char_poolLoss[idx][idy], char_poolmergeLoss[idx][idy]);
        }

        //char pooling
        for (int idy = 0; idy < word_num; idy++) {
          if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
            pool_backward(char_poolLoss[idx][idy][0], char_poolIndex[idx][idy][0], char_hiddenLoss[idx][idy]);
          }
          if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
            pool_backward(char_poolLoss[idx][idy][1], char_poolIndex[idx][idy][1], char_hiddenLoss[idx][idy]);
          }
          if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
            pool_backward(char_poolLoss[idx][idy][2], char_poolIndex[idx][idy][2], char_hiddenLoss[idx][idy]);
          }
        }

        //char convolution
        for (int idy = 0; idy < word_num; idy++) {
          if (idx == seq_size - 1) {
            _char_cnn_project.ComputeBackwardLoss(char_midhidden[idx][idy], char_hidden[idx][idy], char_hiddenLoss[idx][idy], char_midhiddenLoss[idx][idy]);
          } else {
            _hchar_cnn_project.ComputeBackwardLoss(char_midhidden[idx][idy], char_hidden[idx][idy], char_hiddenLoss[idx][idy], char_midhiddenLoss[idx][idy]);
          }
        }

        //char RNN
        for (int idy = 0; idy < word_num; idy++) {
          int char_num = feature.chars[idy].size();
          for (int idz = 0; idz < char_num; idz++) {
            unconcat(char_rnn_hidden_leftLoss[idx][idy][idz], char_rnn_hidden_rightLoss[idx][idy][idz], char_midhiddenLoss[idx][idy][idz]);
          }
        }

        for (int idy = 0; idy < word_num; idy++) {
          if (idx == seq_size - 1) {
            _char_rnn_left.ComputeBackwardLoss(char_input[idx][idy], char_rnn_hidden_left[idx][idy], char_rnn_hidden_leftLoss[idx][idy],
                char_inputLoss[idx][idy]);
            _char_rnn_right.ComputeBackwardLoss(char_input[idx][idy], char_rnn_hidden_right[idx][idy], char_rnn_hidden_rightLoss[idx][idy],
                char_inputLoss[idx][idy]);
          } else {
            _hchar_rnn_left.ComputeBackwardLoss(char_input[idx][idy], char_rnn_hidden_left[idx][idy], char_rnn_hidden_leftLoss[idx][idy],
                char_inputLoss[idx][idy]);
            _hchar_rnn_right.ComputeBackwardLoss(char_input[idx][idy], char_rnn_hidden_right[idx][idy], char_rnn_hidden_rightLoss[idx][idy],
                char_inputLoss[idx][idy]);
          }
        }

        for (int idy = 0; idy < word_num; idy++) {
          windowlized_backward(charprimeLoss[idx][idy], char_inputLoss[idx][idy], char_curcontext);
        }

        if (idx == seq_size - 1) {
          if (_chars.bEmbFineTune()) {
            for (int idy = 0; idy < word_num; idy++) {
              int char_num = feature.chars[idy].size();
              for (int idz = 0; idz < char_num; idz++) {
                charprimeLoss[idx][idy][idz] = charprimeLoss[idx][idy][idz] * charprimeMask[idx][idy][idz];
                _chars.EmbLoss(chars[idy][idz], charprimeLoss[idx][idy][idz]);
              }
            }
          }
        } else {
          if (_hchars.bEmbFineTune()) {
            for (int idy = 0; idy < word_num; idy++) {
              int char_num = feature.chars[idy].size();
              for (int idz = 0; idz < char_num; idz++) {
                charprimeLoss[idx][idy][idz] = charprimeLoss[idx][idy][idz] * charprimeMask[idx][idy][idz];
                _hchars.EmbLoss(chars[idy][idz], charprimeLoss[idx][idy][idz]);
              }
            }
          }
        }

      }

      //release
      for (int idx = 0; idx < seq_size; idx++) {
        //char
        int word_num = example.m_features[idx].words.size();
        for (int idy = 0; idy < word_num; idy++) {
          FreeSpace(&(charprime[idx][idy]));
          FreeSpace(&(charprimeLoss[idx][idy]));
          FreeSpace(&(charprimeMask[idx][idy]));
          FreeSpace(&(char_input[idx][idy]));
          FreeSpace(&(char_inputLoss[idx][idy]));
          FreeSpace(&(char_rnn_hidden_left[idx][idy]));
          FreeSpace(&(char_rnn_hidden_leftLoss[idx][idy]));
          FreeSpace(&(char_rnn_hidden_right[idx][idy]));
          FreeSpace(&(char_rnn_hidden_rightLoss[idx][idy]));
          FreeSpace(&(char_midhidden[idx][idy]));
          FreeSpace(&(char_midhiddenLoss[idx][idy]));
          FreeSpace(&(char_hidden[idx][idy]));
          FreeSpace(&(char_hiddenLoss[idx][idy]));
          for (int idm = 0; idm < _poolmanners; idm++) {
            FreeSpace(&(char_pool[idx][idy][idm]));
            FreeSpace(&(char_poolLoss[idx][idy][idm]));
            FreeSpace(&(char_poolIndex[idx][idy][idm]));
          }

          FreeSpace(&(char_poolmerge[idx][idy]));
          FreeSpace(&(char_poolmergeLoss[idx][idy]));
        }

        FreeSpace(&(wordprime[idx]));
        FreeSpace(&(wordprimeLoss[idx]));
        FreeSpace(&(wordprimeMask[idx]));
        FreeSpace(&(wordrepresent[idx]));
        FreeSpace(&(wordrepresentLoss[idx]));
        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));
        FreeSpace(&(hidden[idx]));
        FreeSpace(&(hiddenLoss[idx]));

        offset = idx * _poolmanners;
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(pool[offset + idm]));
          FreeSpace(&(poolLoss[offset + idm]));
          FreeSpace(&(poolIndex[offset + idm]));
        }
      }

      FreeSpace(&rnn_hidden_left);
      FreeSpace(&rnn_hidden_leftLoss);

      FreeSpace(&rnn_hidden_right);
      FreeSpace(&rnn_hidden_rightLoss);

      FreeSpace(&midhidden);
      FreeSpace(&midhiddenLoss);

      FreeSpace(&poolmerge);
      FreeSpace(&poolmergeLoss);
      FreeSpace(&project);
      FreeSpace(&projectLoss);
      FreeSpace(&output);
      FreeSpace(&outputLoss);
    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  int predict(const vector<int>& linears, const vector<Feature>& features, vector<dtype>& results) {
    int seq_size = features.size();
    int offset = 0;
    if (seq_size > 2) {
      std::cout << "error" << std::endl;
    }

    //char define

    vector<vector<Tensor<xpu, 3, dtype> > > charprime;
    vector<vector<Tensor<xpu, 3, dtype> > > char_input;
    vector<vector<Tensor<xpu, 3, dtype> > > char_rnn_hidden_left;
    vector<vector<Tensor<xpu, 3, dtype> > > char_rnn_hidden_right;
    vector<vector<Tensor<xpu, 3, dtype> > > char_midhidden;
    vector<vector<Tensor<xpu, 3, dtype> > > char_hidden;
    vector<vector<vector<Tensor<xpu, 2, dtype> > > > char_pool;
    vector<vector<vector<Tensor<xpu, 3, dtype> > > > char_poolIndex;
    vector<vector<Tensor<xpu, 2, dtype> > > char_poolmerge;

    charprime.resize(seq_size);
    char_input.resize(seq_size);
    char_rnn_hidden_left.resize(seq_size);
    char_rnn_hidden_right.resize(seq_size);
    char_midhidden.resize(seq_size);
    char_hidden.resize(seq_size);
    char_pool.resize(seq_size);
    char_poolIndex.resize(seq_size);
    char_poolmerge.resize(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      int word_num = features[idx].words.size();
      charprime[idx].resize(word_num);
      char_input[idx].resize(word_num);
      char_rnn_hidden_left[idx].resize(word_num);
      char_rnn_hidden_right[idx].resize(word_num);
      char_midhidden[idx].resize(word_num);
      char_hidden[idx].resize(word_num);
      char_pool[idx].resize(word_num);
      char_poolIndex[idx].resize(word_num);
      char_poolmerge[idx].resize(word_num);
      for (int idy = 0; idy < word_num; idy++) {
        char_pool[idx][idy].resize(_poolmanners);
        char_poolIndex[idx][idy].resize(_poolmanners);
      }

    }

    //word
    Tensor<xpu, 3, dtype> wordprime[seq_size];
    Tensor<xpu, 3, dtype> wordrepresent[seq_size];
    Tensor<xpu, 3, dtype> input[seq_size];

    Tensor<xpu, 3, dtype> rnn_hidden_left;
    Tensor<xpu, 3, dtype> rnn_hidden_right;

    Tensor<xpu, 3, dtype> midhidden;

    Tensor<xpu, 3, dtype> hidden[seq_size];
    vector<Tensor<xpu, 2, dtype> > pool(seq_size * _poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(seq_size * _poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> project;
    Tensor<xpu, 2, dtype> output;

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      int word_num = feature.words.size();

      //char initial
      int char_cnn_iSize = _char_cnn_iSize;
      int charHiddenSize = _charHiddenSize;
      for (int idy = 0; idy < word_num; idy++) {
        int char_num = feature.chars[idy].size();
        charprime[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);

        char_input[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, char_cnn_iSize), d_zero);
        char_rnn_hidden_left[idx] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
        char_rnn_hidden_right[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
        char_midhidden[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, 2 * charHiddenSize), d_zero);
        char_hidden[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);

        for (int idz = 0; idz < _poolmanners; idz++) {
          char_pool[idx][idy][idz] = NewTensor<xpu>(Shape2(1, charHiddenSize), d_zero);
          char_poolIndex[idx][idy][idz] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
        }

        char_poolmerge[idx][idy] = NewTensor<xpu>(Shape2(1, _poolmanners * charHiddenSize), d_zero);
      }

      //word
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

      wordprime[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
      wordrepresent[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
      input[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);

      if (idx == seq_size - 1) {
        rnn_hidden_left = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

        midhidden = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);
      }

      hidden[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
      offset = idx * _poolmanners;
      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
        poolIndex[offset + idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
      }
    }

    poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), d_zero);
    project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      int word_num = feature.words.size();
      int char_num = -1;

      //char
      int window_char = 2 * _charcontext + 1;
      int char_curcontext = _charcontext;
      const vector<vector<int> >& chars = feature.chars;

      //get charEmb
      for (int idy = 0; idy < word_num; idy++) {
        char_num = feature.chars[idy].size();
        for (int idz = 0; idz < char_num; idz++) {
          if (idx == seq_size - 1) {
            _chars.GetEmb(chars[idy][idz], charprime[idx][idy][idz]);
          } else {
            _hchars.GetEmb(chars[idy][idz], charprime[idx][idy][idz]);
          }
        }
      }

      for (int idy = 0; idy < word_num; idy++) {
        windowlized(charprime[idx][idy], char_input[idx][idy], char_curcontext);
      }

      //char RNN
      for (int idy = 0; idy < word_num; idy++) {
        if (idx == seq_size - 1) {
          _char_rnn_left.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_left[idx][idy]);
          _char_rnn_right.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_right[idx][idy]);
        } else {
          _hchar_rnn_left.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_left[idx][idy]);
          _hchar_rnn_right.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_right[idx][idy]);

        }
      }

      for (int idy = 0; idy < word_num; idy++) {
        char_num = feature.chars[idy].size();
        for (int idz = 0; idz < char_num; idz++) {
          concat(char_rnn_hidden_left[idx][idy][idz], char_rnn_hidden_right[idx][idy][idz], char_midhidden[idx][idy][idz]);
        }
      }

      //char convolution

      for (int idy = 0; idy < word_num; idy++) {
        if (idx == seq_size - 1) {
          _char_cnn_project.ComputeForwardScore(char_midhidden[idx][idy], char_hidden[idx][idy]);
        } else {
          _hchar_cnn_project.ComputeForwardScore(char_midhidden[idx][idy], char_hidden[idx][idy]);
        }
      }

      //char pooling
      for (int idy = 0; idy < word_num; idy++) {
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(char_hidden[idx][idy], char_pool[idx][idy][0], char_poolIndex[idx][idy][0]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(char_hidden[idx][idy], char_pool[idx][idy][1], char_poolIndex[idx][idy][1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(char_hidden[idx][idy], char_pool[idx][idy][2], char_poolIndex[idx][idy][2]);
        }
      }

      //charConcat2word
      for (int idy = 0; idy < word_num; idy++) {
        concat(char_pool[idx][idy], char_poolmerge[idx][idy]);
      }

      //word
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

      const vector<int>& words = feature.words;
      //linear features should not be dropped out

      for (int idy = 0; idy < word_num; idy++) {
        if (idx == seq_size - 1)
          _words.GetEmb(words[idy], wordprime[idx][idy]);
        else
          _hwords.GetEmb(words[idy], wordprime[idx][idy]);
      }
      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        //wordrepresent[idx][idy] += wordprime[idx][idy];
        //char_poolmerge[idy] = 0.0;
        concat(wordprime[idx][idy], char_poolmerge[idx][idy], wordrepresent[idx][idy]);
      }

      windowlized(wordrepresent[idx], input[idx], curcontext);

      if (idx == seq_size - 1) {
        _rnn_left.ComputeForwardScore(input[idx], rnn_hidden_left);
        _rnn_right.ComputeForwardScore(input[idx], rnn_hidden_right);

        for (int idy = 0; idy < word_num; idy++) {
          concat(rnn_hidden_left[idy], rnn_hidden_right[idy], midhidden[idy]);
        }

        _cnn_project.ComputeForwardScore(midhidden, hidden[idx]);
      } else {
        for (int idy = 0; idy < word_num; idy++) {
          hidden[idx][idy] += input[idx][idy];
        }
      }

      //word pooling
      offset = idx * _poolmanners;
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(hidden[idx], pool[offset], poolIndex[offset]);
      }
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(hidden[idx], pool[offset + 1], poolIndex[offset + 1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden[idx], pool[offset + 2], poolIndex[offset + 2]);
      }
    }

    // sentence
    offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
    concat(pool, poolmerge, offset);
    _tanh_project.ComputeForwardScore(poolmerge, project);
    _olayer_linear.ComputeForwardScore(project, output);

    // decode algorithm
    int optLabel = softmax_predict(output, results);

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      //char
      int word_num = features[idx].words.size();
      for (int idy = 0; idy < word_num; idy++) {
        FreeSpace(&(charprime[idx][idy]));
        FreeSpace(&(char_input[idx][idy]));
        FreeSpace(&(char_rnn_hidden_left[idx][idy]));
        FreeSpace(&(char_rnn_hidden_right[idx][idy]));
        FreeSpace(&(char_midhidden[idx][idy]));
        FreeSpace(&(char_hidden[idx][idy]));
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(char_pool[idx][idy][idm]));
          FreeSpace(&(char_poolIndex[idx][idy][idm]));
        }

        FreeSpace(&(char_poolmerge[idx][idy]));
      }

      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));

      FreeSpace(&rnn_hidden_left);
      FreeSpace(&rnn_hidden_right);

      FreeSpace(&midhidden);

      FreeSpace(&(hidden[idx]));
      offset = idx * _poolmanners;
      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[offset + idm]));
        FreeSpace(&(poolIndex[offset + idm]));
      }
    }
    FreeSpace(&poolmerge);
    FreeSpace(&project);
    FreeSpace(&output);

    return optLabel;
  }

  dtype computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;
    if (seq_size > 2) {
      std::cout << "error" << std::endl;
    }

    //char define

    vector<vector<Tensor<xpu, 3, dtype> > > charprime;
    vector<vector<Tensor<xpu, 3, dtype> > > char_input;
    vector<vector<Tensor<xpu, 3, dtype> > > char_rnn_hidden_left;
    vector<vector<Tensor<xpu, 3, dtype> > > char_rnn_hidden_right;
    vector<vector<Tensor<xpu, 3, dtype> > > char_midhidden;
    vector<vector<Tensor<xpu, 3, dtype> > > char_hidden;
    vector<vector<vector<Tensor<xpu, 2, dtype> > > > char_pool;
    vector<vector<vector<Tensor<xpu, 3, dtype> > > > char_poolIndex;
    vector<vector<Tensor<xpu, 2, dtype> > > char_poolmerge;

    charprime.resize(seq_size);
    char_input.resize(seq_size);
    char_rnn_hidden_left.resize(seq_size);
    char_rnn_hidden_right.resize(seq_size);
    char_midhidden.resize(seq_size);
    char_hidden.resize(seq_size);
    char_pool.resize(seq_size);
    char_poolIndex.resize(seq_size);
    char_poolmerge.resize(seq_size);

    for (int idx = 0; idx < seq_size; idx++) {
      int word_num = example.m_features[idx].words.size();
      charprime[idx].resize(word_num);
      char_input[idx].resize(word_num);
      char_rnn_hidden_left[idx].resize(word_num);
      char_rnn_hidden_right[idx].resize(word_num);
      char_midhidden[idx].resize(word_num);
      char_hidden[idx].resize(word_num);
      char_pool[idx].resize(word_num);
      char_poolIndex[idx].resize(word_num);
      char_poolmerge[idx].resize(word_num);
      for (int idy = 0; idy < word_num; idy++) {
        char_pool[idx][idy].resize(_poolmanners);
        char_poolIndex[idx][idy].resize(_poolmanners);
      }

    }

    //word
    Tensor<xpu, 3, dtype> wordprime[seq_size];
    Tensor<xpu, 3, dtype> wordrepresent[seq_size];
    Tensor<xpu, 3, dtype> input[seq_size];

    Tensor<xpu, 3, dtype> rnn_hidden_left;
    Tensor<xpu, 3, dtype> rnn_hidden_right;

    Tensor<xpu, 3, dtype> midhidden;

    Tensor<xpu, 3, dtype> hidden[seq_size];
    vector<Tensor<xpu, 2, dtype> > pool(seq_size * _poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(seq_size * _poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> project;
    Tensor<xpu, 2, dtype> output;

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      int word_num = feature.words.size();

      //char initial
      int char_cnn_iSize = _char_cnn_iSize;
      int charHiddenSize = _charHiddenSize;
      for (int idy = 0; idy < word_num; idy++) {
        int char_num = feature.chars[idy].size();
        charprime[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);

        char_input[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, char_cnn_iSize), d_zero);
        char_rnn_hidden_left[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
        char_rnn_hidden_right[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
        char_midhidden[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, 2 * charHiddenSize), d_zero);
        char_hidden[idx][idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);

        for (int idz = 0; idz < _poolmanners; idz++) {
          char_pool[idx][idy][idz] = NewTensor<xpu>(Shape2(1, charHiddenSize), d_zero);
          char_poolIndex[idx][idy][idz] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
        }

        char_poolmerge[idx][idy] = NewTensor<xpu>(Shape2(1, _poolmanners * charHiddenSize), d_zero);
      }

      //word
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

      wordprime[idx] = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
      wordrepresent[idx] = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
      input[idx] = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);

      if (idx == seq_size - 1) {
        rnn_hidden_left = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);
        rnn_hidden_right = NewTensor<xpu>(Shape3(word_num, 1, _wordHiddenSize), d_zero);

        midhidden = NewTensor<xpu>(Shape3(word_num, 1, 2 * _wordHiddenSize), d_zero);
      }

      hidden[idx] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
      offset = idx * _poolmanners;
      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[offset + idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
        poolIndex[offset + idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
      }
    }

    poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize + _poolmanners * _token_representation_size), d_zero);
    project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      int word_num = feature.words.size();
      int char_num = -1;

      //char
      int window_char = 2 * _charcontext + 1;
      int char_curcontext = _charcontext;
      const vector<vector<int> >& chars = feature.chars;

      //get charEmb
      for (int idy = 0; idy < word_num; idy++) {
        char_num = feature.chars[idy].size();
        for (int idz = 0; idz < char_num; idz++) {
          if (idx == seq_size - 1) {
            _chars.GetEmb(chars[idy][idz], charprime[idx][idy][idz]);
          } else {
            _hchars.GetEmb(chars[idy][idz], charprime[idx][idy][idz]);
          }
        }
      }

      for (int idy = 0; idy < word_num; idy++) {
        windowlized(charprime[idx][idy], char_input[idx][idy], char_curcontext);
      }

      //char RNN
      for (int idy = 0; idy < word_num; idy++) {
        if (idx == seq_size - 1) {
          _char_rnn_left.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_left[idx][idy]);
          _char_rnn_right.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_right[idx][idy]);
        } else {
          _hchar_rnn_left.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_left[idx][idy]);
          _hchar_rnn_right.ComputeForwardScore(char_input[idx][idy], char_rnn_hidden_right[idx][idy]);

        }
      }

      for (int idy = 0; idy < word_num; idy++) {
        char_num = feature.chars[idy].size();
        for (int idz = 0; idz < char_num; idz++) {
          concat(char_rnn_hidden_left[idx][idy][idz], char_rnn_hidden_right[idx][idy][idz], char_midhidden[idx][idy][idz]);
        }
      }

      //char convolution
      for (int idy = 0; idy < word_num; idy++) {
        if (idx == seq_size - 1) {
          _char_cnn_project.ComputeForwardScore(char_midhidden[idx][idy], char_hidden[idx][idy]);
        } else {
          _hchar_cnn_project.ComputeForwardScore(char_midhidden[idx][idy], char_hidden[idx][idy]);
        }
      }

      //char pooling
      for (int idy = 0; idy < word_num; idy++) {
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(char_hidden[idx][idy], char_pool[idx][idy][0], char_poolIndex[idx][idy][0]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(char_hidden[idx][idy], char_pool[idx][idy][1], char_poolIndex[idx][idy][1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(char_hidden[idx][idy], char_pool[idx][idy][2], char_poolIndex[idx][idy][2]);
        }
      }

      //charConcat2word
      for (int idy = 0; idy < word_num; idy++) {
        concat(char_pool[idx][idy], char_poolmerge[idx][idy]);
      }

      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

      const vector<int>& words = feature.words;
      //linear features should not be dropped out

      for (int idy = 0; idy < word_num; idy++) {
        if (idx == seq_size - 1)
          _words.GetEmb(words[idy], wordprime[idx][idy]);
        else
          _hwords.GetEmb(words[idy], wordprime[idx][idy]);
      }
      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        //wordrepresent[idx][idy] += wordprime[idx][idy];
        //char_poolmerge[idy] = 0.0;
        concat(wordprime[idx][idy], char_poolmerge[idx][idy], wordrepresent[idx][idy]);
      }

      windowlized(wordrepresent[idx], input[idx], curcontext);

      if (idx == seq_size - 1) {
        _rnn_left.ComputeForwardScore(input[idx], rnn_hidden_left);
        _rnn_right.ComputeForwardScore(input[idx], rnn_hidden_right);

        for (int idy = 0; idy < word_num; idy++) {
          concat(rnn_hidden_left[idy], rnn_hidden_right[idy], midhidden[idy]);
        }

        _cnn_project.ComputeForwardScore(midhidden, hidden[idx]);
      } else {
        for (int idy = 0; idy < word_num; idy++) {
          hidden[idx][idy] += input[idx][idy];
        }
      }

      //word pooling
      offset = idx * _poolmanners;
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(hidden[idx], pool[offset], poolIndex[offset]);
      }
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(hidden[idx], pool[offset + 1], poolIndex[offset + 1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden[idx], pool[offset + 2], poolIndex[offset + 2]);
      }
    }

    // sentence
    offset = (seq_size == 1) ? (_poolmanners * _token_representation_size) : 0;
    concat(pool, poolmerge, offset);
    _tanh_project.ComputeForwardScore(poolmerge, project);
    _olayer_linear.ComputeForwardScore(project, output);

    dtype cost = softmax_cost(output, example.m_labels);

    //release
    //release
    for (int idx = 0; idx < seq_size; idx++) {
      //char
      int word_num = example.m_features[idx].words.size();
      for (int idy = 0; idy < word_num; idy++) {
        FreeSpace(&(charprime[idx][idy]));
        FreeSpace(&(char_input[idx][idy]));
        FreeSpace(&(char_rnn_hidden_left[idx][idy]));
        FreeSpace(&(char_rnn_hidden_right[idx][idy]));
        FreeSpace(&(char_midhidden[idx][idy]));
        FreeSpace(&(char_hidden[idx][idy]));
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(char_pool[idx][idy][idm]));
          FreeSpace(&(char_poolIndex[idx][idy][idm]));
        }

        FreeSpace(&(char_poolmerge[idx][idy]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));

      FreeSpace(&rnn_hidden_left);
      FreeSpace(&rnn_hidden_right);

      FreeSpace(&midhidden);

      FreeSpace(&(hidden[idx]));
      offset = idx * _poolmanners;
      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[offset + idm]));
        FreeSpace(&(poolIndex[offset + idm]));
      }
    }
    FreeSpace(&poolmerge);
    FreeSpace(&project);
    FreeSpace(&output);

    return cost;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    //char
    _char_cnn_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _char_rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _char_rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _chars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _hchar_cnn_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _hchar_rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _hchar_rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _hchars.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    //word
    _cnn_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _hwords.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(this, examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(this, examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    checkgrad(this, examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(this, examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    checkgrad(this, examples, _cnn_project._W, _cnn_project._gradW, "_cnn_project._W", iter);
    checkgrad(this, examples, _cnn_project._b, _cnn_project._gradb, "_cnn_project._b", iter);

    checkgrad(this, examples, _rnn_left._rnn._WL, _rnn_left._rnn._gradWL, "_rnn_left._rnn._WL", iter);
    checkgrad(this, examples, _rnn_left._rnn._WR, _rnn_left._rnn._gradWR, "_rnn_left._rnn._WR", iter);
    checkgrad(this, examples, _rnn_left._rnn._b, _rnn_left._rnn._gradb, "_rnn_left._rnn._b", iter);

    checkgrad(this, examples, _rnn_right._rnn._WL, _rnn_right._rnn._gradWL, "_rnn_right._rnn._WL", iter);
    checkgrad(this, examples, _rnn_right._rnn._WR, _rnn_right._rnn._gradWR, "_rnn_right._rnn._WR", iter);
    checkgrad(this, examples, _rnn_right._rnn._b, _rnn_right._rnn._gradb, "_rnn_right._rnn._b", iter);

    checkgrad(this, examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
    checkgrad(this, examples, _hwords._E, _hwords._gradE, "_hwords._E", iter, _hwords._indexers);
    //char
    checkgrad(this, examples, _char_cnn_project._W, _char_cnn_project._gradW, "_char_cnn_project._W", iter);
    checkgrad(this, examples, _char_cnn_project._b, _char_cnn_project._gradb, "_char_cnn_project._b", iter);

    checkgrad(this, examples, _char_rnn_left._rnn._WL, _char_rnn_left._rnn._gradWL, "_char_rnn_left._rnn._WL", iter);
    checkgrad(this, examples, _char_rnn_left._rnn._WR, _char_rnn_left._rnn._gradWR, "_char_rnn_left._rnn._WR", iter);
    checkgrad(this, examples, _char_rnn_left._rnn._b, _char_rnn_left._rnn._gradb, "_char_rnn_left._rnn._b", iter);

    checkgrad(this, examples, _char_rnn_right._rnn._WL, _char_rnn_right._rnn._gradWL, "_char_rnn_right._rnn._WL", iter);
    checkgrad(this, examples, _char_rnn_right._rnn._WR, _char_rnn_right._rnn._gradWR, "_char_rnn_right._rnn._WR", iter);
    checkgrad(this, examples, _char_rnn_right._rnn._b, _char_rnn_right._rnn._gradb, "_char_rnn_right._rnn._b", iter);

    checkgrad(this, examples, _chars._E, _chars._gradE, "_chars._E", iter, _chars._indexers);

    checkgrad(this, examples, _hchar_cnn_project._W, _hchar_cnn_project._gradW, "_hchar_cnn_project._W", iter);
    checkgrad(this, examples, _hchar_cnn_project._b, _hchar_cnn_project._gradb, "_hchar_cnn_project._b", iter);

    checkgrad(this, examples, _hchar_rnn_left._rnn._WL, _hchar_rnn_left._rnn._gradWL, "_hchar_rnn_left._rnn._WL", iter);
    checkgrad(this, examples, _hchar_rnn_left._rnn._WR, _hchar_rnn_left._rnn._gradWR, "_hchar_rnn_left._rnn._WR", iter);
    checkgrad(this, examples, _hchar_rnn_left._rnn._b, _hchar_rnn_left._rnn._gradb, "_hchar_rnn_left._rnn._b", iter);

    checkgrad(this, examples, _hchar_rnn_right._rnn._WL, _hchar_rnn_right._rnn._gradWL, "_hchar_rnn_right._rnn._WL", iter);
    checkgrad(this, examples, _hchar_rnn_right._rnn._WR, _hchar_rnn_right._rnn._gradWR, "_hchar_rnn_right._rnn._WR", iter);
    checkgrad(this, examples, _hchar_rnn_right._rnn._b, _hchar_rnn_right._rnn._gradb, "_hchar_rnn_right._rnn._b", iter);

    checkgrad(this, examples, _hchars._E, _hchars._gradE, "_hchars._E", iter, _hchars._indexers);
  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
    _words.setEmbFineTune(b_wordEmb_finetune);
  }

  inline void setCharEmbFinetune(bool b_charEmb_finetune) {
    _chars.setEmbFineTune(b_charEmb_finetune);
  }

  inline void resetRemove(int remove) {
    _remove = remove;
  }

};

#endif /* SRC_RCNNHSWordCharClassifier_H_ */
