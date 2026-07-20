// Copyright (c) MediaTek Inc. All rights reserved.
// Licensed under the MIT License.
//
// Neuron execution provider option flags.
// These constants are used by NeuronExecutionProvider to interpret the
// provider_options map passed at session creation.

#pragma once

#include <stdint.h>

static const uint32_t NEURON_FLAG_USE_FP16         = 1 << 0;  // Run inference in FP16
static const uint32_t NEURON_FLAG_USE_NCHW         = 1 << 1;  // Force NCHW layout
static const uint32_t NEURON_FLAG_CPU_DISABLED      = 1 << 2;  // Disable CPU fallback
static const uint32_t NEURON_FLAG_CPU_ONLY          = 1 << 3;  // Force CPU backend only
static const uint32_t NEURON_FLAG_MDLA_ONLY         = 1 << 4;  // Target MDLA accelerator only
