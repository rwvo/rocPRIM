// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common_test_header.hpp"

// required rocprim headers
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>

// required test headers
#include "test_utils.hpp"
#include "test_utils_types.hpp"

template<
    class T,
    class U,
    unsigned int ItemsPerThread,
    bool ShouldBeVectorized
>
struct params
{
    using type = T;
    using vector_type = U;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr bool should_be_vectorized = ShouldBeVectorized;
};

template<
    class Type,
    rocprim::block_load_method Load,
    rocprim::block_store_method Store,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
struct class_params
{
    using type = Type;
    static constexpr rocprim::block_load_method load_method = Load;
    static constexpr rocprim::block_store_method store_method = Store;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

template<class ClassParams>
class RocprimBlockLoadStoreClassTests : public ::testing::Test {
public:
    using params = ClassParams;
};

template<class Params>
class RocprimVectorizationTests : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    // block_load_direct
    /*class_params<int, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 512U, 3>,
    class_params<int, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 512U, 4>,
    class_params<int, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 512U, 5>,
    class_params<int, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 512U, 6>,*/


    //class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_direct,
    //             rocprim::block_store_method::block_store_direct, 256U, 4>,
    //class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_direct,
    //             rocprim::block_store_method::block_store_direct, 256U, 5>,
    //class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_direct,
    //             rocprim::block_store_method::block_store_direct, 256U, 6>,
    class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_direct,
                 rocprim::block_store_method::block_store_direct, 256U, 7>,



    // block_load_vectorize
    class_params<double, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 512U, 2>,

    //class_params<double, rocprim::block_load_method::block_load_vectorize,
    //             rocprim::block_store_method::block_store_vectorize, 512U, 3>,

    class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_vectorize,
                 rocprim::block_store_method::block_store_vectorize, 256U, 4>/*,

    // block_load_transpose

    class_params<double, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 512U, 3>,*/


    /*class_params<test_utils::custom_test_type<double>, rocprim::block_load_method::block_load_transpose,
                 rocprim::block_store_method::block_store_transpose, 256U, 4>*/

> ClassParams;

TYPED_TEST_CASE(RocprimBlockLoadStoreClassTests, ClassParams);

template<
    class Type,
    rocprim::block_load_method LoadMethod,
    rocprim::block_store_method StoreMethod,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
void load_store_kernel(Type* device_input, Type* device_output)
{
    Type items[ItemsPerThread];
    unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread;
    rocprim::block_load<Type, BlockSize, ItemsPerThread, LoadMethod> load;
    rocprim::block_store<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.load(device_input + offset, items);
    store.store(device_output + offset, items);
}

TYPED_TEST(RocprimBlockLoadStoreClassTests, LoadStoreClass)
{
    int device_id = test_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Type = typename TestFixture::params::type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr rocprim::block_load_method load_method = TestFixture::params::load_method;
    constexpr rocprim::block_store_method store_method = TestFixture::params::store_method;
    const size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_block;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100, seed_value);
        std::vector<Type> output(input.size(), 0);

        // Calculate expected results on host
        std::vector<Type> expected(input.size(), 0);
        for (size_t i = 0; i < 113; i++)
        {
            size_t block_offset = i * items_per_block;
            for (size_t j = 0; j < items_per_block; j++)
            {
                expected[j + block_offset] = input[j + block_offset];
            }
        }

        // Preparing device
        Type* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        Type* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(typename decltype(input)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                load_store_kernel<
                    Type, load_method, store_method,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output
        );

        // Reading results from device
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }

}
