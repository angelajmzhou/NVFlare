//
//  ETTrainer.mm
//  NVFlareMobile
//
//

#include "ETTrainer.h"
#import <UIKit/UIKit.h>
#import <executorch/extension/data_loader/file_data_loader.h>
#import <executorch/extension/module/module.h>
#import <executorch/extension/tensor/tensor.h>
#import <executorch/extension/training/module/training_module.h>
#import <executorch/extension/training/optimizer/sgd.h>
#import <os/log.h>
#import <os/signpost.h>

// WORKAROUND: Include implementation files directly because training libraries
// are missing from the framework
#ifdef __cplusplus
#include <executorch/extension/training/module/training_module.cpp>
#include <executorch/extension/training/optimizer/sgd.cpp>
#endif
#import "../Bridge/NVFlareConstants.h"
#include "../Bridge/SwiftDatasetAdapter.h"
#include "ETDataset.h"
#include "ETDebugUtils.h"
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ::executorch::extension;

@implementation ETTrainer {
  std::unique_ptr<training::TrainingModule> _training_module;
  NSDictionary<NSString *, id> *_meta;
  ETDataset *_dataset; // Non-owning raw pointer. The dataset object's lifetime
                       // must exceed that of the ETTrainer instance. The caller
                       // is responsible for ensuring the dataset remains valid
                       // for the duration of training.
}

// loadDataset method removed - dataset passed directly to initializer

- (std::unique_ptr<training::TrainingModule>)loadModel:(NSString *)modelBase64 {
  // Decode base64 string to temporary file
  NSData *modelData = [[NSData alloc] initWithBase64EncodedString:modelBase64
                                                          options:0];
  if (!modelData) {
    return nullptr;
  }

  // Write to temporary file
  NSString *tempPath =
      [NSTemporaryDirectory() stringByAppendingPathComponent:@"temp_model.pte"];
  if (![modelData writeToFile:tempPath atomically:YES]) {
    return nullptr;
  }

  std::unique_ptr<training::TrainingModule> module;
  @try {
    // Load model using FileDataLoader
    auto model_result = FileDataLoader::from(tempPath.UTF8String);
    if (!model_result.ok()) {
      return nullptr;
    }

    auto loader =
        std::make_unique<FileDataLoader>(std::move(model_result.get()));
    module = std::make_unique<training::TrainingModule>(std::move(loader));

  } @catch (NSException *exception) {
    module = nullptr;
  }

  // Clean up temporary file
  [[NSFileManager defaultManager] removeItemAtPath:tempPath error:nil];

  return module;
}

/// Primary initializer - accepts C++ dataset directly
- (instancetype)initWithModelBase64:(NSString *)modelBase64
                               meta:(NSDictionary<NSString *, id> *)meta
                            dataset:(void *)cppDataset {
  NSLog(@"ETTrainer: Initialization started with app's C++ dataset");
  self = [super init];
  if (self) {
    _meta = meta;

    // Use app's C++ dataset directly
    if (!cppDataset) {
      NSLog(@"ETTrainer: App provided null C++ dataset");
      return nil;
    }

    NSLog(@"ETTrainer: cppDataset pointer = %p", cppDataset);

    // Cast to our expected C++ dataset type
    // Note: cppDataset comes from SwiftDatasetBridge which creates
    // SwiftDatasetAdapter (inherits from ETDataset)
    ETDataset *dataset = static_cast<ETDataset *>(cppDataset);

    // Basic validation - ensure the pointer makes sense
    if (!dataset) {
      NSLog(@"ETTrainer: dataset pointer is null after cast");
      return nil;
    }

    NSLog(@"ETTrainer: After successful cast, dataset pointer = %p", dataset);

    // Store raw pointer (non-owning)
    _dataset = dataset;

    NSLog(
        @"ETTrainer: Dataset pointer stored, about to test object validity...");

    // Test if the pointer is valid before calling methods
    if (!_dataset) {
      NSLog(@"ETTrainer: Dataset pointer is null after creation");
      return nil;
    }

    NSLog(@"ETTrainer: Dataset pointer = %p", _dataset);

    // Safely try to access the dataset
    @try {
      NSLog(@"ETTrainer: About to call size() on dataset...");
      size_t datasetSize = _dataset->size();
      NSLog(@"ETTrainer: App's C++ dataset ready (size: %zu)", datasetSize);
    } @catch (NSException *exception) {
      NSLog(@"ETTrainer: NSException accessing dataset: %@", exception);
      return nil;
    } @catch (...) {
      NSLog(@"ETTrainer: C++ exception accessing dataset");
      return nil;
    }

    // Load model
    NSLog(@"ETTrainer: Loading ExecutorTorch model");
    _training_module = [self loadModel:modelBase64];
    if (!_training_module) {
      NSLog(@"ETTrainer: Failed to load ExecutorTorch model");
      return nil;
    }
    NSLog(@"ETTrainer: ExecutorTorch model loaded successfully");
  }
  NSLog(@"ETTrainer: Initialization complete with app's C++ dataset");
  return self;
}

// Helper methods removed - app provides C++ dataset directly

+ (NSDictionary<NSString *, id> *)toTensorDictionary:
    (const std::map<executorch::aten::string_view, executorch::aten::Tensor> &)
        map {
  NSMutableDictionary *tensorDict = [NSMutableDictionary dictionary];

  for (const auto &pair : map) {
    NSString *fullKey = [NSString stringWithUTF8String:pair.first.data()];
    // FIX: Strip "net." prefix to match server expectation
    NSString *key = [fullKey stringByReplacingOccurrencesOfString:@"net."
                                                       withString:@""];
    executorch::aten::Tensor tensor = pair.second;

    NSMutableDictionary *singleTensorDict = [NSMutableDictionary dictionary];

    auto strides = tensor.strides();
    auto sizes = tensor.sizes();
    auto data_ptr = tensor.const_data_ptr<float>();

    NSMutableArray *stridesArray =
        [NSMutableArray arrayWithCapacity:strides.size()];
    for (size_t i = 0; i < strides.size(); ++i) {
      [stridesArray addObject:@(strides[i])];
    }
    singleTensorDict[@"strides"] = stridesArray;

    NSMutableArray *sizesArray =
        [NSMutableArray arrayWithCapacity:sizes.size()];
    for (size_t i = 0; i < sizes.size(); ++i) {
      [sizesArray addObject:@(sizes[i])];
    }
    singleTensorDict[@"sizes"] = sizesArray;

    NSMutableArray *dataArray =
        [NSMutableArray arrayWithCapacity:tensor.numel()];
    for (size_t i = 0; i < tensor.numel(); ++i) {
      [dataArray addObject:@(data_ptr[i])];
    }
    singleTensorDict[@"data"] = dataArray;

    tensorDict[key] = singleTensorDict;
  }

  return tensorDict;
}

+ (NSDictionary<NSString *, id> *)
    calculateTensorDifference:(NSDictionary<NSString *, id> *)oldDict
                      newDict:(NSDictionary<NSString *, id> *)newDict {
  NSMutableDictionary *diffDict = [NSMutableDictionary dictionary];

  for (NSString *key in oldDict) {
    NSDictionary *oldTensor = oldDict[key];
    NSDictionary *newTensor = newDict[key];

    if (!newTensor) {
      NSLog(@"Warning: Tensor %@ not found in new parameters", key);
      continue;
    }

    NSArray *oldData = oldTensor[@"data"];
    NSArray *newData = newTensor[@"data"];

    if (oldData.count != newData.count) {
      NSLog(@"Warning: Tensor %@ size mismatch: old=%lu new=%lu", key,
            (unsigned long)oldData.count, (unsigned long)newData.count);
      continue;
    }

    NSMutableArray *diffData = [NSMutableArray arrayWithCapacity:oldData.count];
    for (NSUInteger i = 0; i < oldData.count; i++) {
      float oldVal = [oldData[i] floatValue];
      float newVal = [newData[i] floatValue];
      float diff = newVal - oldVal;
      [diffData addObject:@(diff)];
    }

    NSMutableDictionary *diffTensor = [NSMutableDictionary dictionary];
    diffTensor[@"sizes"] = oldTensor[@"sizes"];     // Keep original sizes
    diffTensor[@"strides"] = oldTensor[@"strides"]; // Keep original strides
    diffTensor[@"data"] = diffData;                 // Store differences

    diffDict[key] = diffTensor;
  }

  return diffDict;
}

- (NSDictionary<NSString *, id> *)train {
  NSLog(@"ETTrainer: Starting train()");
  if (!_training_module) {
    NSLog(@"ETTrainer: Training module not initialized");
    return @{};
  }

  @try {
    int batchSize = [_meta[kNVFlareMetaKeyBatchSize] intValue];
    NSLog(@"ETTrainer: Using batch size: %d", batchSize);

    // Run dummy forward/backward to initialize parameters
    NSLog(@"ETTrainer: Running dummy forward pass to initialize parameters");
    _dataset->reset();
    auto dummyBatch = _dataset->getBatch(batchSize);
    if (!dummyBatch) {
      NSLog(@"ETTrainer: Failed to get dummy batch!");
      return @{};
    }
    const auto &[dummyInput, dummyLabel] = *dummyBatch;

    // Check batch size
    size_t dummyBatchActualSize = dummyInput->sizes()[0];
    if (dummyBatchActualSize != batchSize) {
      NSLog(@"ETTrainer: Warning - dummy batch size %zu != requested %d",
            dummyBatchActualSize, batchSize);
    }

    // Dynamic method name resolution
    std::string methodName;
    auto methodsRes = _training_module->method_names();
    if (methodsRes.ok()) {
      const auto &names = methodsRes.get();
      if (names.empty()) {
        NSString *msg = @"ETTrainer: ERROR - No method names found in module!";
        NSLog(@"%@", msg);
        return @{@"_error" : msg};
      }

      // Default to picking 'forward' if available
      bool found = false;
      for (const auto &name : names) {
        if (std::string(name) == "forward") {
          methodName = "forward";
          found = true;
          break;
        }
      }

      // Fallback: Pick the first available method
      if (!found) {
        methodName = *names.begin();
        NSLog(@"ETTrainer: 'forward' method not found! Defaulting to first "
              @"available method: %s",
              methodName.c_str());
      } else {
        NSLog(@"ETTrainer: Using standard 'forward' method.");
      }
    } else {
      NSString *msg =
          [NSString stringWithFormat:@"ETTrainer: Failed to get method names: "
                                     @"%d. Defaulting to 'forward'",
                                     (int)methodsRes.error()];
      NSLog(@"%@", msg);

      // Still try to continue with 'forward' but log the warning to the result
      // if it fails later? For now, let's just log and continue as before, but
      // if method_names failed, execution likely will too.
      methodName = "forward";
    }

    auto dummyRes = _training_module->execute_forward_backward(
        methodName.c_str(), {*dummyInput, *dummyLabel});
    if (dummyRes.error() != executorch::runtime::Error::Ok) {
      NSString *msg = [NSString
          stringWithFormat:
              @"ETTrainer: Dummy pass failed! Error code: %d. Method: %s",
              (int)dummyRes.error(), methodName.c_str()];
      NSLog(@"%@", msg);
      return @{@"_error" : msg};
    }
    _dataset->reset(); // Reset for actual training

    // Get initial parameters
    auto param_res = _training_module->named_parameters(methodName.c_str());
    if (param_res.error() != executorch::runtime::Error::Ok) {
      NSString *msg = @"ETTrainer: Failed to get named parameters";
      NSLog(@"%@", msg);
      return @{@"_error" : msg};
    }

    auto initial_params = param_res.get();
    NSDictionary<NSString *, id> *old_params =
        [ETTrainer toTensorDictionary:initial_params];
    NSLog(@"ETTrainer: Got initial parameters");

    printTensorDictionary(old_params, @"Initial Params");

    // Configure optimizer
    float learningRate = [_meta[kNVFlareMetaKeyLearningRate] floatValue];
    training::optimizer::SGDOptions options{learningRate};
    training::optimizer::SGD optimizer(param_res.get(), options);

    // Train the model
    NSInteger totalEpochs = [_meta[kNVFlareMetaKeyTotalEpochs] integerValue];
    int totalSteps = 0;
    size_t datasetSize = _dataset->size();
    if (datasetSize < (size_t)batchSize) {
      NSString *msg = [NSString
          stringWithFormat:@"ETTrainer: ERROR - Dataset too small for batch "
                           @"size! Dataset size: %zu, Batch size: %d",
                           datasetSize, batchSize];
      NSLog(@"%@", msg);
      return @{@"_error" : msg};
    }
    size_t numBatchesPerEpoch =
        datasetSize / batchSize; // Floor division - drop incomplete batches
    size_t samplesUsedPerEpoch = numBatchesPerEpoch * batchSize;
    size_t droppedSamples = datasetSize - samplesUsedPerEpoch;

    NSLog(@"ETTrainer: Dataset size: %zu, Batch size: %d", datasetSize,
          batchSize);
    NSLog(@"ETTrainer: Batches per epoch: %zu, Samples used: %zu, Dropped: %zu",
          numBatchesPerEpoch, samplesUsedPerEpoch, droppedSamples);

    if (numBatchesPerEpoch == 0) {
      NSString *msg = [NSString
          stringWithFormat:@"ETTrainer: ERROR - Dataset too small (0 batches)! "
                           @"Dataset size: %zu, Batch size: %d",
                           datasetSize, batchSize];
      NSLog(@"%@", msg);
      return @{@"_error" : msg};
    }

    for (int epoch = 0; epoch < totalEpochs; epoch++) {
      _dataset->reset(); // Reset dataset at the start of each epoch
      size_t epochSamplesProcessed =
          0; // Track samples processed in current epoch

      os_log_t log = os_log_create("com.nvidia.nvflare.ios", "Training");
      os_signpost_id_t spid = os_signpost_id_generate(log);

      for (size_t batchIdx = 0; batchIdx < numBatchesPerEpoch; batchIdx++) {
        os_signpost_interval_begin(log, spid, "TrainingStep",
                                   "Epoch %d Batch %zu", epoch, batchIdx);

        auto batchOpt = _dataset->getBatch(batchSize);
        if (!batchOpt) {
          os_signpost_interval_end(log, spid, "TrainingStep");
          break; // End of dataset
        }

        const auto &[input, label] = *batchOpt;

        // Ensure fixed batch size - drop incomplete batches
        size_t actualBatchSize = input->sizes()[0];
        if (actualBatchSize != batchSize) {
          NSLog(@"Dropping incomplete batch: expected %d samples, got %zu "
                @"samples",
                batchSize, actualBatchSize);
          os_signpost_interval_end(log, spid, "TrainingStep");
          break; // Skip remaining incomplete batches in this epoch
        }
        const auto &results = _training_module->execute_forward_backward(
            methodName.c_str(), {*input, *label});

        if (results.error() != executorch::runtime::Error::Ok) {
          NSString *msg = [NSString
              stringWithFormat:@"Failed to execute forward_backward (Error %d)",
                               (int)results.error()];
          NSLog(@"%@", msg);
          os_signpost_interval_end(log, spid, "TrainingStep");
          return @{@"_error" : msg};
        }

        // Track samples processed (all batches are now fixed size)
        epochSamplesProcessed += batchSize;

        if (totalSteps % 500 == 0 ||
            (epoch == totalEpochs - 1 && batchIdx == numBatchesPerEpoch - 1)) {
          NSLog(@"Epoch %d/%lld, Progress %.1f%%, Step %d, Loss %f, Prediction "
                @"%lld, Label %lld",
                epoch + 1, (long long)totalEpochs,
                (float)epochSamplesProcessed * 100 / samplesUsedPerEpoch,
                totalSteps,
                results.get()[0].toTensor().const_data_ptr<float>()[0],
                results.get()[1].toTensor().const_data_ptr<int64_t>()[0],
                label->const_data_ptr<int64_t>()[0]);
        }

        optimizer.step(
            _training_module->named_gradients(methodName.c_str()).get());
        totalSteps++;

        os_signpost_interval_end(log, spid, "TrainingStep");
      }
    }

    NSDictionary<NSString *, id> *final_params =
        [ETTrainer toTensorDictionary:param_res.get()];

    printTensorDictionary(final_params, @"Final Params");

    auto tensor_diff = [ETTrainer calculateTensorDifference:old_params
                                                    newDict:final_params];

    printTensorDictionary(tensor_diff, @"Tensor Diff");

    return tensor_diff;

  } @catch (NSException *exception) {
    NSString *msg = [NSString
        stringWithFormat:@"Training failed with Exception: %@", exception];
    NSLog(@"%@", msg);
    return @{@"_error" : msg};
  }
}

@end
