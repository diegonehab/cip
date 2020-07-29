// Copyright 2012--2020 Leonardo Sacht, Rodolfo Schulz de Lima, Diego Nehab
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#ifndef GPU_HOSTDEV_H
#define GPU_HOSTDEV_H

#if !defined(HOSTDEV)
#   if defined(__CUDA_ARCH__)
#       define HOSTDEV __host__ __device__
#   else
#       define HOSTDEV
#   endif
#endif

#endif
