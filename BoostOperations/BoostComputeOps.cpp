//
// Created by hkumar on 26.08.20.
//

#pragma once
#include "BoostComputeOps.h"


compute::vector<int> BoostCompute::getBoostGpuData(vector<int> data, compute::context context, compute::command_queue queue) {

    compute::vector<int> device_vector(data.size(),context);
    compute::copy(data.begin(), data.end(), device_vector.begin(), queue);
    return device_vector;
}

vector<int> BoostCompute::getBoostCpuData(compute::vector<int> deviceData,compute::command_queue queue) {

    vector<int> hostData(deviceData.size());
    compute::copy(deviceData.begin(),deviceData.end(),hostData.begin(),queue);
    return hostData;
}

vector<float> BoostCompute::getBoostCpuData(compute::vector<float> deviceData,compute::command_queue queue) {

    vector<float> hostData(deviceData.size());
    compute::copy(deviceData.begin(),deviceData.end(),hostData.begin(),queue);
    return hostData;
}


compute::vector<int> BoostCompute::boostSelection(compute::vector<int> deviceVec,
                                                  string operation,
                                                  int value,
                                                  compute::command_queue queue,
                                                  compute::context context,
                                                  compute::device device) {

    compute::vector<int> deviceResult(deviceVec.size(),context);

    BOOST_COMPUTE_CLOSURE(int,greaterOrEqualTo,(int x),(value),{
        return ( x >= value);
    });

    BOOST_COMPUTE_CLOSURE(int, lesserOrEqualTo,(int x),(value),{
        return ( x <= value);
    });

    BOOST_COMPUTE_CLOSURE(int,greaterThan,(int x),(value),{
        return ( x > value);
    });

    BOOST_COMPUTE_CLOSURE(int,lesserThan,(int x),(value),{
        return ( x < value);
    });

    BOOST_COMPUTE_CLOSURE(int,equalTo,(int x),(value),{
        return ( x == value);
    });

    BOOST_COMPUTE_CLOSURE(int,notEqual,(int x),(value),{
        return ( x != value);
    });


    if(!operation.compare("GE")) {
        compute::transform(deviceVec.begin(), deviceVec.end(), deviceResult.begin(),
                           greaterOrEqualTo, queue);
    }
    else if(!operation.compare("LE")) {
        compute::transform(deviceVec.begin(), deviceVec.end(), deviceResult.begin(),
                           lesserOrEqualTo, queue);
    }
    else if(!operation.compare("G")) {
        compute::transform(deviceVec.begin(), deviceVec.end(), deviceResult.begin(),
                           greaterThan, queue);
    }
    else if(!operation.compare("L")) {
        compute::transform(deviceVec.begin(), deviceVec.end(), deviceResult.begin(),
                           lesserThan, queue);
    }
    else if(!operation.compare("EQ")) {
        compute::transform(deviceVec.begin(), deviceVec.end(), deviceResult.begin(),
                           equalTo, queue);
    }
    else{
        compute::transform(deviceVec.begin(), deviceVec.end(), deviceResult.begin(),
                           notEqual, queue);
    }

    return deviceResult;
}

compute::vector<int> BoostCompute::boostSelectionArrays(compute::vector<int> lhs, string operation,
                                                        compute::vector<int> rhs, compute::command_queue queue,
                                                        compute::context context) {

    compute::vector<int> deviceResult(lhs.size(),context);
//    size_t v1size = lhs.size();

    BOOST_COMPUTE_CLOSURE(int,greaterEqual,(std::size_t x),(lhs,rhs,deviceResult),{
                deviceResult[x] = (lhs[x] >= rhs[x]);
    });

    BOOST_COMPUTE_CLOSURE(int,lesserEqual,(std::size_t x),(lhs,rhs,deviceResult),{
        deviceResult[x] = (lhs[x] <= rhs[x]);
    });

    BOOST_COMPUTE_CLOSURE(int,greater,(std::size_t x),(lhs,rhs,deviceResult),{
        deviceResult[x] = (lhs[x] > rhs[x]);
    });

    BOOST_COMPUTE_CLOSURE(int,lesser,(std::size_t x),(lhs,rhs,deviceResult),{
        deviceResult[x] = (lhs[x] < rhs[x]);
    });

    BOOST_COMPUTE_CLOSURE(int,Equal,(std::size_t x),(lhs,rhs,deviceResult),{
        deviceResult[x] = (lhs[x] == rhs[x]);
    });

    BOOST_COMPUTE_CLOSURE(int,notEqual,(std::size_t x),(lhs,rhs,deviceResult),{
        deviceResult[x] = (lhs[x] != rhs[x]);
    });

    if(!operation.compare("GE")) {
        compute::for_each_n(compute::counting_iterator<int>(0), lhs.size(), greaterEqual, queue);
    }
    else if(!operation.compare("LE")) {
        compute::for_each_n(compute::counting_iterator<int>(0), lhs.size(), lesserEqual, queue);
    }
    else if(!operation.compare("G")) {
        compute::for_each_n(compute::counting_iterator<int>(0), lhs.size(), greater, queue);
    }
    else if(!operation.compare("L")) {
        compute::for_each_n(compute::counting_iterator<int>(0), lhs.size(), lesser, queue);
    }
    else if(!operation.compare("EQ")) {
        compute::for_each_n(compute::counting_iterator<int>(0), lhs.size(), Equal, queue);
    }
    else{
        compute::for_each_n(compute::counting_iterator<int>(0), lhs.size(), notEqual, queue);
    }

    return deviceResult;

}

compute::vector<int> BoostCompute::boostConjunction(compute::vector<int> deviceLHS,
                                                    compute::vector<int> deviceRHS,
                                                    compute::command_queue queue,
                                                    compute::context context) {

    compute::vector<int> deviceResult(deviceLHS.size(),context);

    compute::transform(deviceLHS.begin(),
                       deviceLHS.end(),
                       deviceRHS.begin(),
                       deviceResult.begin(),
                       compute::bit_and<int>(),
                       queue);
    return deviceResult;
}

compute::vector<int> BoostCompute::boostJoin(compute::vector<int> deviceParent, compute::vector<int> deviceChild,
                                             compute::command_queue queue, compute::context context) {

    compute::vector<int> deviceResult(deviceChild.size(),context);
    size_t v1size = deviceParent.size();

    BOOST_COMPUTE_CLOSURE(int,nestedLoop,(std::size_t x),(deviceParent,deviceChild,deviceResult,v1size),{
        deviceResult[x] = -1;
        for(int i = 0; i < v1size; i++){
            if(deviceChild[x] == deviceParent[i]){
                deviceResult[x] = i;
//                return i;
            }
        }
    });

    compute::for_each_n(compute::counting_iterator<int>(0),deviceChild.size(),nestedLoop,queue);
    return deviceResult;
}

compute::vector<int> BoostCompute::boostProduct(compute::vector<int> deviceLHS, compute::vector<int> deviceRHS,
                                                compute::command_queue queue, compute::context context) {

    compute::vector<int> deviceResult(deviceLHS.size(),context);

    compute::transform(deviceLHS.begin(),
                       deviceLHS.end(),
                       deviceRHS.begin(),
                       deviceResult.begin(),
                       compute::multiplies<int>(),
                       queue);

    return deviceResult;
}

int BoostCompute::boostSum(compute::vector<int> deviceData, compute::command_queue queue,
                           compute::context context) {
    compute::vector<int> deviceResult(deviceData.size(),context);

// reason to use reduce not accumulate - because the former is faster
//    https://www.boost.org/doc/libs/1_65_1/libs/compute/doc/html/boost/compute/accumulate.html
    compute::reduce(deviceData.begin(), deviceData.end(),deviceResult.begin(),compute::plus<int>(),queue);
    int result = deviceResult[0];
    return result;
}

float BoostCompute::boostAvg(compute::vector<int> deviceData, compute::command_queue queue, compute::context context) {
    compute::vector<float> deviceResult(deviceData.size(),context);
    compute::reduce(deviceData.begin(), deviceData.end(),deviceResult.begin(),compute::plus<int>(),queue);
    int total = deviceResult[0];
    float result = (float)total/deviceData.size();
    return result;
}

int BoostCompute::boostCountIf(compute::vector<int> deviceData, compute::command_queue queue, compute::context context,
                               int value) {
    return compute::count(deviceData.begin(),deviceData.end(),value,queue);
}

int BoostCompute::boostCount(compute::vector<int> deviceData) {
    return deviceData.size();
}

compute::vector<int> BoostCompute::boostPrefixSum(compute::vector<int> deviceSelData, compute::command_queue queue,
                                                  compute::context context) {
    compute::vector<int> dataIndex(deviceSelData.size(),context);
    compute::iota(dataIndex.begin(),dataIndex.end(),0,queue);

    compute::vector<int> ps(deviceSelData.size(),context);
    compute::exclusive_scan(deviceSelData.begin(),deviceSelData.end(),ps.begin(),queue); // prefix sum result
//
    int size = compute::count(deviceSelData.begin(),deviceSelData.end(),1,queue);
    compute::vector<int> device_result(size,context);

    compute::scatter_if(dataIndex.begin(),dataIndex.end(),ps.begin(),deviceSelData.begin(),device_result.begin(),queue);

    return device_result;
}

int BoostCompute::boostFindMax(compute::vector<int> data, compute::command_queue queue, compute::context context) {
    compute::vector<float> deviceResult(data.size(),context);
    compute::reduce(data.begin(), data.end(),deviceResult.begin(),compute::max<int>(),queue);
    return deviceResult[0];
}

int BoostCompute::boostFindMin(compute::vector<int> data, compute::command_queue queue, compute::context context) {
    compute::vector<int> deviceResult(data.size(),context);
    compute::reduce(data.begin(), data.end(),deviceResult.begin(),compute::min<int>(),queue);
    return deviceResult[0];
//    compute::vector<int>::iterator it = compute::min_element(data.begin(),data.end());
//    int index = compute::distance<int>(data.begin(),it);
//    return data[index];
}

compute::vector<int> BoostCompute::boostSort(compute::vector<int> deviceData, int order, compute::command_queue queue,
                                             compute::context context) {
    if(order){
        compute::sort(deviceData.begin(),deviceData.end(),compute::greater<int>(),queue);
    }else{
        compute::sort(deviceData.begin(),deviceData.end(),queue);
    }
    return deviceData;
}

compute::vector<int> BoostCompute::boostSortByKey(compute::vector<int> deviceData,
                                                  compute::vector<int> dependentData, int order,
                                                  compute::command_queue queue,
                                                  compute::context context) {

//    compute::vector<int> index(deviceData.size(),context);
//    compute::iota(index.begin(), index.end(), 0,queue);

    // sort index based on ref_data
    if (order) {
        compute::sort_by_key(deviceData.begin(), deviceData.end(), dependentData.begin(),compute::greater<int>(), queue);
    }else{
        compute::sort_by_key(deviceData.begin(), deviceData.end(), dependentData.begin(), queue);
    }

//    BOOST_COMPUTE_CLOSURE(int,sortAscending,(std::size_t x, std::size_t y),(deviceData),{
//        return deviceData[x] < deviceData[y];
//    });
//
//    BOOST_COMPUTE_CLOSURE(int,sortDescending,(std::size_t x, std::size_t y),(deviceData),{
//        return deviceData[x] > deviceData[y];
//    });
//
//
//    if(order){
//        compute::sort(index.begin(),index.end(),sortDescending,queue);
//    }else{
//        compute::sort(index.begin(),index.end(),sortAscending,queue);
//    }
//
//    compute::vector<int> deviceResult(dependentData.size(),context);
//    compute::gather(index.begin(),index.end(),dependentData.begin(),deviceResult.begin(),queue);
//    return deviceResult;
    return dependentData;
}

compute::vector<int> BoostCompute::boostGroupBy(compute::vector<int> keys, compute::vector<int> values, compute::command_queue queue,
                                                compute::context context) {

    compute::vector<int> temp = keys;
    auto last = compute::unique(temp.begin(),temp.end(),queue);
    temp.erase(last, temp.end());
    int size = temp.size();

    compute::vector<int> oKeys(size,context);
    compute::vector<int> oVals(size,context);

    compute::reduce_by_key(keys.begin(), keys.end(), values.begin(), oKeys.begin(), oVals.begin(),queue);
    return oVals;
}

compute::vector<int> BoostCompute::boostCountByKey(compute::vector<int> data, compute::command_queue queue,
                                                   compute::context context) {

    compute::vector<int> temp = data;
    auto last = compute::unique(temp.begin(),temp.end(),queue);
    temp.erase(last, temp.end());
    int size = temp.size();

    compute::vector<int> C(size,context);
    compute::vector<int> D(size,context);

    compute::reduce_by_key(data.begin(),data.end(), compute::make_constant_iterator(1),
                           C.begin(),D.begin(),compute::plus<int>(),
                           compute::equal_to<int>(),queue);

    return D;
}

compute::vector<float> BoostCompute::boostAvgByKey(compute::vector<int> k, compute::vector<int> v,
                                                 compute::command_queue queue, compute::context context) {

    compute::vector<int> temp = k;
    auto last = compute::unique(temp.begin(),temp.end(),queue);
    temp.erase(last, temp.end());
    int size = temp.size();

    compute::vector<int> keys(size,context);
    compute::vector<int> sums(size,context);
    compute::vector<int> counts(size,context);
    compute::vector<float> result(size,context);

    compute::reduce_by_key(k.begin(), k.end(), v.begin(), keys.begin(), sums.begin(),queue);
    compute::reduce_by_key(k.begin(),k.end(), compute::make_constant_iterator(1),
                           keys.begin(),counts.begin(),compute::plus<int>(),
                           compute::equal_to<int>(),queue);

    BOOST_COMPUTE_CLOSURE(int,average,(std::size_t x),(sums,counts,result),{
        result[x] = (float)(sums[x] / counts[x]);
    });

    compute::for_each_n(compute::counting_iterator<int>(0), keys.size(),average,queue);

    return result;
}

compute::vector<int> BoostCompute::boostSumOfVectors(compute::vector<int> array1, compute::vector<int> array2,
                                                     compute::command_queue queue, compute::context context) {

    compute::vector<int> deviceResult(array1.size(),context);

    BOOST_COMPUTE_CLOSURE(int,sum,(std::size_t x),(array1,array2,deviceResult),{
        deviceResult[x] = (array1[x] + array2[x]);
    });

    compute::for_each_n(compute::counting_iterator<int>(0),array1.size(),sum,queue);

    return deviceResult;

}