/*
 * CNNWordCharClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_CNNWordCharClassifier_H_
#define SRC_CNNWordCharClassifier_H_

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
class CNNWordCharClassifier {
public:
  CNNWordCharClassifier() {
    _dropOut = 0.5;
  }
  ~CNNWordCharClassifier() {

  }

public:
  LookupTable<xpu> _words;

  int _wordcontext;
  int _wordSize;
  int _wordDim;
  bool _b_wordEmb_finetune;
  int _wordHiddenSize;
  int _word_cnn_iSize;

  //char
  LookupTable<xpu> _chars;
  int _charcontext;
  int _charSize;
  int _charDim;
  bool _b_charEmb_finetune;
  int _charHiddenSize;
  int _char_cnn_iSize;

  int _token_representation_size;

  int _hiddenSize;

  UniLayer<xpu> _cnn_project;
  UniLayer<xpu> _tanh_project;
  UniLayer<xpu> _olayer_linear;

  UniLayer<xpu> _char_cnn_project;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3 min

  int _poolmanners;

public:

  inline void init(const NRMat<dtype>& wordEmb, const NRMat<dtype>& charEmb, int wordcontext, int charcontext, int labelSize, int wordHiddenSize,
      int charHiddenSize, int hiddenSize) {
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
    _char_cnn_project.initial(_charHiddenSize, _char_cnn_iSize, true, 40, 0);

    _labelSize = labelSize;
    _hiddenSize = hiddenSize;
    _wordHiddenSize = wordHiddenSize;
    _token_representation_size = _wordDim + _poolmanners * _charHiddenSize;

    _word_cnn_iSize = _token_representation_size * (2 * _wordcontext + 1);

    _words.initial(wordEmb);

    _cnn_project.initial(_wordHiddenSize, _word_cnn_iSize, true, 30, 0);
    _tanh_project.initial(_hiddenSize, _poolmanners * _wordHiddenSize, true, 50, 0);
    _olayer_linear.initial(_labelSize, _hiddenSize, false, 60, 2);

    _eval.reset();

    _remove = 0;

  }

  inline void release() {
    _words.release();
    _cnn_project.release();
    _tanh_project.release();
    _olayer_linear.release();

    _chars.release();
    _char_cnn_project.release();
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

      int idx = seq_size - 1;

      int word_num = example.m_features[idx].words.size();

      //char
      vector<Tensor<xpu, 3, dtype> > charprime(word_num), charprimeLoss(word_num), charprimeMask(word_num);
      vector<Tensor<xpu, 3, dtype> > char_input(word_num), char_inputLoss(word_num);
      vector<Tensor<xpu, 3, dtype> > char_hidden(word_num), char_hiddenLoss(word_num);
      vector<vector<Tensor<xpu, 2, dtype> > > char_pool(word_num, vector<Tensor<xpu, 2, dtype> >(_poolmanners)), char_poolLoss(word_num,
          vector<Tensor<xpu, 2, dtype> >(_poolmanners));
      vector<vector<Tensor<xpu, 3, dtype> > > char_poolIndex(word_num, vector<Tensor<xpu, 3, dtype> >(_poolmanners));
      vector<Tensor<xpu, 2, dtype> > char_poolmerge(word_num), char_poolmergeLoss(word_num);

      //word
      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> wordrepresent, wordrepresentLoss;
      Tensor<xpu, 3, dtype> input, inputLoss;
      Tensor<xpu, 3, dtype> hidden, hiddenLoss;
      vector<Tensor<xpu, 2, dtype> > pool(_poolmanners), poolLoss(_poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

      Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, dtype> project, projectLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;

      //initialize
      idx = seq_size - 1;

      {
        const Feature& feature = example.m_features[idx];
        int word_num = feature.words.size();

        //char initial
        int char_cnn_iSize = _char_cnn_iSize;
        int charHiddenSize = _charHiddenSize;
        for (int idy = 0; idy < word_num; idy++) {
          int char_num = feature.chars[idy].size();
          charprime[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);
          charprimeLoss[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);
          charprimeMask[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_one);

          char_input[idy] = NewTensor<xpu>(Shape3(char_num, 1, char_cnn_iSize), d_zero);
          char_inputLoss[idy] = NewTensor<xpu>(Shape3(char_num, 1, char_cnn_iSize), d_zero);
          char_hidden[idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
          char_hiddenLoss[idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);

          for (int idz = 0; idz < _poolmanners; idz++) {
            char_pool[idy][idz] = NewTensor<xpu>(Shape2(1, charHiddenSize), d_zero);
            char_poolLoss[idy][idz] = NewTensor<xpu>(Shape2(1, charHiddenSize), d_zero);
            char_poolIndex[idy][idz] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
          }

          char_poolmerge[idy] = NewTensor<xpu>(Shape2(1, _poolmanners * charHiddenSize), d_zero);
          char_poolmergeLoss[idy] = NewTensor<xpu>(Shape2(1, _poolmanners * charHiddenSize), d_zero);
        }

        //word initial
        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
        int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

        wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
        wordprimeLoss = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
        wordprimeMask = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_one);
        wordrepresent = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
        wordrepresentLoss = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);

        input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);
        inputLoss = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);
        hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
        hiddenLoss = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

        for (int idm = 0; idm < _poolmanners; idm++) {
          pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
          poolLoss[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
          poolIndex[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
        }
      }

      poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
      project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
      projectLoss = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
      output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

      //forward propagation
      //input setting, and linear setting
      {
        const Feature& feature = example.m_features[idx];
        int word_num = feature.words.size();
        int char_num = -1;

        //char
        int window_char = 2 * _charcontext + 1;
        int char_curcontext = _charcontext;
        const vector<vector<int> >& chars = feature.chars;
        srand(iter * example_num + count * seq_size + idx);

        //get charEmb
        for (int idy = 0; idy < word_num; idy++) {
          char_num = feature.chars[idy].size();
          for (int idz = 0; idz < char_num; idz++) {
            _chars.GetEmb(chars[idy][idz], charprime[idy][idz]);
          }
        }

        //char dropout
        for (int idy = 0; idy < word_num; idy++) {
          char_num = feature.chars[idy].size();
          for (int idz = 0; idz < char_num; idz++) {
            dropoutcol(charprimeMask[idy][idz], _dropOut);
            charprime[idy][idz] = charprime[idy][idz] * charprimeMask[idy][idz];
          }
        }

        for (int idy = 0; idy < word_num; idy++) {
          windowlized(charprime[idy], char_input[idy], char_curcontext);
        }

        //char convolution

        for (int idy = 0; idy < word_num; idy++) {
          _char_cnn_project.ComputeForwardScore(char_input[idy], char_hidden[idy]);
        }

        //char pooling
        for (int idy = 0; idy < word_num; idy++) {
          if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
            avgpool_forward(char_hidden[idy], char_pool[idy][0], char_poolIndex[idy][0]);
          }
          if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
            maxpool_forward(char_hidden[idy], char_pool[idy][1], char_poolIndex[idy][1]);
          }
          if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
            minpool_forward(char_hidden[idy], char_pool[idy][2], char_poolIndex[idy][2]);
          }
        }

        //charConcat2word
        for (int idy = 0; idy < word_num; idy++) {
          concat(char_pool[idy], char_poolmerge[idy]);
        }

        //word
        int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
        int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

        const vector<int>& words = feature.words;
        word_num = words.size();
        //linear features should not be dropped out

        for (int idy = 0; idy < word_num; idy++) {
          _words.GetEmb(words[idy], wordprime[idy]);
        }

        //word dropout
        for (int idy = 0; idy < word_num; idy++) {
          dropoutcol(wordprimeMask[idy], _dropOut);
          wordprime[idy] = wordprime[idy] * wordprimeMask[idy];
        }

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          //wordrepresent[idy] += wordprime[idy];
          //char_poolmerge[idy] = 0.0;
          concat(wordprime[idy], char_poolmerge[idy], wordrepresent[idy]);
        }

        windowlized(wordrepresent, input, curcontext);

        //word convolution
        for (int idy = 0; idy < word_num; idy++) {
          _cnn_project.ComputeForwardScore(input[idy], hidden[idy]);
        }

        //word pooling
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(hidden, pool[0], poolIndex[0]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(hidden, pool[1], poolIndex[1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(hidden, pool[2], poolIndex[2]);
        }
      }

      // sentence
      concat(pool, poolmerge);
      _tanh_project.ComputeForwardScore(poolmerge, project);
      _olayer_linear.ComputeForwardScore(project, output);

      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      //sentence
      _olayer_linear.ComputeBackwardLoss(project, output, outputLoss, projectLoss);
      _tanh_project.ComputeBackwardLoss(poolmerge, project, projectLoss, poolmergeLoss);

      unconcat(poolLoss, poolmergeLoss);

      {

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
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          pool_backward(poolLoss[0], poolIndex[0], hiddenLoss);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          pool_backward(poolLoss[1], poolIndex[1], hiddenLoss);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          pool_backward(poolLoss[2], poolIndex[2], hiddenLoss);
        }

        //word convolution
        for (int idy = 0; idy < word_num; idy++) {
          _cnn_project.ComputeBackwardLoss(input[idy], hidden[idy], hiddenLoss[idy], inputLoss[idy]);
        }

        windowlized_backward(wordrepresentLoss, inputLoss, curcontext);

        //word representation
        for (int idy = 0; idy < word_num; idy++) {
          //wordprimeLoss[idy] += wordrepresentLoss[idy];
          unconcat(wordprimeLoss[idy], char_poolmergeLoss[idy], wordrepresentLoss[idy]);
          //char_poolmergeLoss[idy] = 0.0;
        }

        //dropout
        if (_words.bEmbFineTune()) {
          for (int idy = 0; idy < word_num; idy++) {
            wordprimeLoss[idy] = wordprimeLoss[idy] * wordprimeMask[idy];
            _words.EmbLoss(words[idy], wordprimeLoss[idy]);
          }
        }

        //char loss backward propagation
        for (int idy = 0; idy < word_num; idy++) {
          unconcat(char_poolLoss[idy], char_poolmergeLoss[idy]);
        }

        //char pooling
        for (int idy = 0; idy < word_num; idy++) {
          if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
            pool_backward(char_poolLoss[idy][0], char_poolIndex[idy][0], char_hiddenLoss[idy]);
          }
          if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
            pool_backward(char_poolLoss[idy][1], char_poolIndex[idy][1], char_hiddenLoss[idy]);
          }
          if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
            pool_backward(char_poolLoss[idy][2], char_poolIndex[idy][2], char_hiddenLoss[idy]);
          }
        }

        //char convolution

        for (int idy = 0; idy < word_num; idy++) {
          _char_cnn_project.ComputeBackwardLoss(char_input[idy], char_hidden[idy], char_hiddenLoss[idy], char_inputLoss[idy]);
        }

        for (int idy = 0; idy < word_num; idy++) {
          windowlized_backward(charprimeLoss[idy], char_inputLoss[idy], char_curcontext);
        }

        //dropout
        if (_chars.bEmbFineTune()) {
          for (int idy = 0; idy < word_num; idy++) {
            int char_num = feature.chars[idy].size();
            for (int idz = 0; idz < char_num; idz++) {
              charprimeLoss[idy][idz] = charprimeLoss[idy][idz] * charprimeMask[idy][idz];
              _chars.EmbLoss(chars[idy][idz], charprimeLoss[idy][idz]);
            }
          }
        }

      }

      //release
      {

        //char
        for (int idy = 0; idy < word_num; idy++) {
          FreeSpace(&(charprime[idy]));
          FreeSpace(&(charprimeLoss[idy]));
          FreeSpace(&(charprimeMask[idy]));
          FreeSpace(&(char_input[idy]));
          FreeSpace(&(char_inputLoss[idy]));
          FreeSpace(&(char_hidden[idy]));
          FreeSpace(&(char_hiddenLoss[idy]));
          for (int idm = 0; idm < _poolmanners; idm++) {
            FreeSpace(&(char_pool[idy][idm]));
            FreeSpace(&(char_poolLoss[idy][idm]));
            FreeSpace(&(char_poolIndex[idy][idm]));
          }

          FreeSpace(&(char_poolmerge[idy]));
          FreeSpace(&(char_poolmergeLoss[idy]));
        }

        FreeSpace(&wordprime);
        FreeSpace(&wordprimeLoss);
        FreeSpace(&wordprimeMask);
        FreeSpace(&wordrepresent);
        FreeSpace(&wordrepresentLoss);
        FreeSpace(&input);
        FreeSpace(&inputLoss);
        FreeSpace(&hidden);
        FreeSpace(&hiddenLoss);
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(pool[idm]));
          FreeSpace(&(poolLoss[idm]));
          FreeSpace(&(poolIndex[idm]));
        }
      }
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

    int idx = seq_size - 1;
    int word_num = features[idx].words.size();

    //char
    vector<Tensor<xpu, 3, dtype> > charprime(word_num);
    vector<Tensor<xpu, 3, dtype> > char_input(word_num);
    vector<Tensor<xpu, 3, dtype> > char_hidden(word_num);
    vector<vector<Tensor<xpu, 2, dtype> > > char_pool(word_num, vector<Tensor<xpu, 2, dtype> >(_poolmanners));
    vector<vector<Tensor<xpu, 3, dtype> > > char_poolIndex(word_num, vector<Tensor<xpu, 3, dtype> >(_poolmanners));
    vector<Tensor<xpu, 2, dtype> > char_poolmerge(word_num);

    //word
    Tensor<xpu, 3, dtype> wordprime;
    Tensor<xpu, 3, dtype> wordrepresent;
    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> hidden;
    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> project;
    Tensor<xpu, 2, dtype> output;

    //initialize
    idx = seq_size - 1;

    {

      const Feature& feature = features[idx];
      int word_num = feature.words.size();

      //char initial
      int char_cnn_iSize = _char_cnn_iSize;
      int charHiddenSize = _charHiddenSize;
      for (int idy = 0; idy < word_num; idy++) {
        int char_num = feature.chars[idy].size();
        charprime[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);

        char_input[idy] = NewTensor<xpu>(Shape3(char_num, 1, char_cnn_iSize), d_zero);
        char_hidden[idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);

        for (int idz = 0; idz < _poolmanners; idz++) {
          char_pool[idy][idz] = NewTensor<xpu>(Shape2(1, charHiddenSize), d_zero);
          char_poolIndex[idy][idz] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
        }

        char_poolmerge[idy] = NewTensor<xpu>(Shape2(1, _poolmanners * charHiddenSize), d_zero);
      }

      //word
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

      wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
      wordrepresent = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
      input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);
      hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
        poolIndex[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
      }
    }
    poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
    project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

    //forward propagation
    //input setting, and linear setting
    {
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
          _chars.GetEmb(chars[idy][idz], charprime[idy][idz]);
        }
      }

      for (int idy = 0; idy < word_num; idy++) {
        windowlized(charprime[idy], char_input[idy], char_curcontext);
      }

      //char convolution

      for (int idy = 0; idy < word_num; idy++) {
        _char_cnn_project.ComputeForwardScore(char_input[idy], char_hidden[idy]);
      }

      //char pooling
      for (int idy = 0; idy < word_num; idy++) {
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(char_hidden[idy], char_pool[idy][0], char_poolIndex[idy][0]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(char_hidden[idy], char_pool[idy][1], char_poolIndex[idy][1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(char_hidden[idy], char_pool[idy][2], char_poolIndex[idy][2]);
        }
      }

      //charConcat2word
      for (int idy = 0; idy < word_num; idy++) {
        concat(char_pool[idy], char_poolmerge[idy]);
      }

      //word
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

      const vector<int>& words = feature.words;
      //linear features should not be dropped out

      for (int idy = 0; idy < word_num; idy++) {
        _words.GetEmb(words[idy], wordprime[idy]);
      }

      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        //wordrepresent[idy] +=  wordprime[idy];
        //char_poolmerge[idy] = 0.0;
        concat(wordprime[idy], char_poolmerge[idy], wordrepresent[idy]);
      }

      windowlized(wordrepresent, input, curcontext);

      //word convolution
      for (int idy = 0; idy < word_num; idy++) {
        _cnn_project.ComputeForwardScore(input[idy], hidden[idy]);
      }

      //word pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(hidden, pool[0], poolIndex[0]);
      }
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(hidden, pool[1], poolIndex[1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden, pool[2], poolIndex[2]);
      }
    }

    // sentence
    concat(pool, poolmerge);
    _tanh_project.ComputeForwardScore(poolmerge, project);
    _olayer_linear.ComputeForwardScore(project, output);

    // decode algorithm
    int optLabel = softmax_predict(output, results);

    //release
    {
      //char
      for (int idy = 0; idy < word_num; idy++) {
        FreeSpace(&(charprime[idy]));
        FreeSpace(&(char_input[idy]));
        FreeSpace(&(char_hidden[idy]));
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(char_pool[idy][idm]));
          FreeSpace(&(char_poolIndex[idy][idm]));
        }

        FreeSpace(&(char_poolmerge[idy]));
      }

      FreeSpace(&wordprime);
      FreeSpace(&wordrepresent);
      FreeSpace(&input);
      FreeSpace(&hidden);
      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[idm]));
        FreeSpace(&(poolIndex[idm]));
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

    int idx = seq_size - 1;
    int word_num = example.m_features[idx].words.size();

    //char
    vector<Tensor<xpu, 3, dtype> > charprime(word_num);
    vector<Tensor<xpu, 3, dtype> > char_input(word_num);
    vector<Tensor<xpu, 3, dtype> > char_hidden(word_num);
    vector<vector<Tensor<xpu, 2, dtype> > > char_pool(word_num, vector<Tensor<xpu, 2, dtype> >(_poolmanners));
    vector<vector<Tensor<xpu, 3, dtype> > > char_poolIndex(word_num, vector<Tensor<xpu, 3, dtype> >(_poolmanners));
    vector<Tensor<xpu, 2, dtype> > char_poolmerge(word_num);

    Tensor<xpu, 3, dtype> wordprime;
    Tensor<xpu, 3, dtype> wordrepresent;
    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> hidden;
    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> project;
    Tensor<xpu, 2, dtype> output;

    //initialize
    idx = seq_size - 1;

    {

      const Feature& feature = example.m_features[idx];
      int word_num = feature.words.size();

      //char initial
      int char_cnn_iSize = _char_cnn_iSize;
      int charHiddenSize = _charHiddenSize;
      for (int idy = 0; idy < word_num; idy++) {
        int char_num = feature.chars[idy].size();
        charprime[idy] = NewTensor<xpu>(Shape3(char_num, 1, _charDim), d_zero);

        char_input[idy] = NewTensor<xpu>(Shape3(char_num, 1, char_cnn_iSize), d_zero);
        char_hidden[idy] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);

        for (int idz = 0; idz < _poolmanners; idz++) {
          char_pool[idy][idz] = NewTensor<xpu>(Shape2(1, charHiddenSize), d_zero);
          char_poolIndex[idy][idz] = NewTensor<xpu>(Shape3(char_num, 1, charHiddenSize), d_zero);
        }

        char_poolmerge[idy] = NewTensor<xpu>(Shape2(1, _poolmanners * charHiddenSize), d_zero);
      }

      //word
      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int word_cnn_iSize = (idx == seq_size - 1) ? _word_cnn_iSize : _token_representation_size;
      int wordHiddenSize = (idx == seq_size - 1) ? _wordHiddenSize : _token_representation_size;

      wordprime = NewTensor<xpu>(Shape3(word_num, 1, _wordDim), d_zero);
      wordrepresent = NewTensor<xpu>(Shape3(word_num, 1, _token_representation_size), d_zero);
      input = NewTensor<xpu>(Shape3(word_num, 1, word_cnn_iSize), d_zero);
      hidden = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);

      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[idm] = NewTensor<xpu>(Shape2(1, wordHiddenSize), d_zero);
        poolIndex[idm] = NewTensor<xpu>(Shape3(word_num, 1, wordHiddenSize), d_zero);
      }
    }

    poolmerge = NewTensor<xpu>(Shape2(1, _poolmanners * _wordHiddenSize), d_zero);
    project = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

    //forward propagation
    //input setting, and linear setting
    {

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
          _chars.GetEmb(chars[idy][idz], charprime[idy][idz]);
        }
      }

      for (int idy = 0; idy < word_num; idy++) {
        windowlized(charprime[idy], char_input[idy], char_curcontext);
      }

      //char convolution

      for (int idy = 0; idy < word_num; idy++) {
        _char_cnn_project.ComputeForwardScore(char_input[idy], char_hidden[idy]);
      }

      //char pooling
      for (int idy = 0; idy < word_num; idy++) {
        if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
          avgpool_forward(char_hidden[idy], char_pool[idy][0], char_poolIndex[idy][0]);
        }
        if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
          maxpool_forward(char_hidden[idy], char_pool[idy][1], char_poolIndex[idy][1]);
        }
        if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
          minpool_forward(char_hidden[idy], char_pool[idy][2], char_poolIndex[idy][2]);
        }
      }

      //charConcat2word
      for (int idy = 0; idy < word_num; idy++) {
        concat(char_pool[idy], char_poolmerge[idy]);
      }

      int window = (idx == seq_size - 1) ? (2 * _wordcontext + 1) : 1;
      int curcontext = (idx == seq_size - 1) ? _wordcontext : 0;

      const vector<int>& words = feature.words;
      //linear features should not be dropped out

      for (int idy = 0; idy < word_num; idy++) {
        _words.GetEmb(words[idy], wordprime[idy]);
      }

      //word representation
      for (int idy = 0; idy < word_num; idy++) {
        //wordrepresent[idy] +=  wordprime[idy];
        //char_poolmerge[idy] = 0.0;
        concat(wordprime[idy], char_poolmerge[idy], wordrepresent[idy]);
      }

      windowlized(wordrepresent, input, curcontext);

      //word convolution
      for (int idy = 0; idy < word_num; idy++) {
        _cnn_project.ComputeForwardScore(input[idy], hidden[idy]);
      }

      //word pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(hidden, pool[0], poolIndex[0]);
      }
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(hidden, pool[1], poolIndex[1]);
      }
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(hidden, pool[2], poolIndex[2]);
      }
    }

    // sentence
    concat(pool, poolmerge);
    _tanh_project.ComputeForwardScore(poolmerge, project);
    _olayer_linear.ComputeForwardScore(project, output);

    dtype cost = softmax_cost(output, example.m_labels);

    //release
    {
      //char
      for (int idy = 0; idy < word_num; idy++) {
        FreeSpace(&(charprime[idy]));
        FreeSpace(&(char_input[idy]));
        FreeSpace(&(char_hidden[idy]));
        for (int idm = 0; idm < _poolmanners; idm++) {
          FreeSpace(&(char_pool[idy][idm]));
          FreeSpace(&(char_poolIndex[idy][idm]));
        }

        FreeSpace(&(char_poolmerge[idy]));
      }

      FreeSpace(&wordprime);
      FreeSpace(&wordrepresent);
      FreeSpace(&input);
      FreeSpace(&hidden);
      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[idm]));
        FreeSpace(&(poolIndex[idm]));
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

    _cnn_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _chars.updateAdaGrad(nnRegular, adaAlpha, adaEps);

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

    checkgrad(this, examples, _char_cnn_project._W, _char_cnn_project._gradW, "_char_cnn_project._W", iter);
    checkgrad(this, examples, _char_cnn_project._b, _char_cnn_project._gradb, "_char_cnn_project._b", iter);

    checkgrad(this, examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
    checkgrad(this, examples, _chars._E, _chars._gradE, "_chars._E", iter, _chars._indexers);

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

#endif /* SRC_CNNWordCharClassifier_H_ */
