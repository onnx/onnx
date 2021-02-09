/*
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <string>
#include <unordered_map>
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

class OpSet_Versions {
public:
    void addOpset(std::string opset_name, std::string domain, int version, OpSchema& class_name) {
        // do we need to handle different domains?
        onnx_version_map[opset_name].push_back(version);
        op_class_map[toMapKey(opset_name, domain, version)] = class_name;
    }
    std::vector<OpSchema> getAllLatestVersion(std::string domain, int target_version) {
        if (domain == "Onnx") {
            std::vector<OpSchema> class_names;
            for (auto element : onnx_version_map) {
                class_names.push_back(*getLatestVersionBeforeTarget(element.first, domain, target_version));
            }
            return class_names;
        }
        // TODO: handle other domains
        return {};
    }
private:
    std::unordered_map<std::string, std::vector<int>> onnx_version_map; // for Onnx domain
    std::unordered_map<std::string, OpSchema> op_class_map;
    OpSchema* getLatestVersionBeforeTarget(std::string opset_name, std::string domain, int target_version) {
        for (auto version : onnx_version_map[opset_name]) {
            if (version >= target_version) {
                return &op_class_map[toMapKey(opset_name, domain, version)];
            }
        }
        // cannot find supported opset schema before target version
        return NULL;
    }
    std::string toMapKey(std::string opset_name, std::string domain, int version) {
        return opset_name + "_" + domain + "_" + std::to_string(version);
    }

};
}
