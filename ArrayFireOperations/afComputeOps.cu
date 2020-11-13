//
// Created by hkumar on 31.08.20.
//

#pragma once
#include "afComputeOps.cuh"

af::array afCompute::getAFGpuData(std::vector<int> data) {
    int* hostData = &data[0];
    af::array deviceData((dim_t)data.size(), hostData, afHost);
    return deviceData;
}

af::array afCompute::getAFGpuData(std::vector<float> data) {
    float * hostData = &data[0];
    af::array deviceData((dim_t)data.size(), hostData, afHost);
    return deviceData;
}

vector<int> afCompute::getAFCpuData(af::array deviceData) {
    vector<int> hostData(deviceData.elements());
    deviceData.host(hostData.data());
    return hostData;
}

vector<float> afCompute::getAFFloatCpuData(af::array deviceData) {
    vector<float> hostData(deviceData.elements());
    deviceData.host(hostData.data());
    return hostData;
}

af::array afCompute::afSelection(af::array deviceData, string operation, int value) {

    af::array device_result;

    if(!operation.compare("GE")) {
        device_result = af::where(af::operator>=(deviceData, value));
    }
    else if(!operation.compare("LE")) {

        device_result = af::where(af::operator<=(deviceData, value));
    }
    else if(!operation.compare("G")) {
        device_result = af::where(af::operator>(deviceData, value));
    }
    else if(!operation.compare("L")) {
        device_result = af::where(af::operator<(deviceData, value));
    }
    else if(!operation.compare("EQ")) {
        device_result = af::where(af::operator==(deviceData, value));
    }
    else{
        device_result = af::where(af::operator!=(deviceData, value));
    }

    return device_result;
}

af::array afCompute::afSelectionArrays(af::array lhs, string operation, af::array rhs) {

    af::array device_result;

    if(!operation.compare("GE")) {
        device_result = af::where(af::operator>=(lhs, rhs));
    }
    else if(!operation.compare("LE")) {

        device_result = af::where(af::operator<=(lhs, rhs));
    }
    else if(!operation.compare("G")) {
        device_result = af::where(af::operator>(lhs, rhs));
    }
    else if(!operation.compare("L")) {
        device_result = af::where(af::operator<(lhs, rhs));
    }
    else if(!operation.compare("EQ")) {
        device_result = af::where(af::operator==(lhs, rhs));
    }
    else{
        device_result = af::where(af::operator!=(lhs, rhs));
    }

    return device_result;
}

af::array afCompute::afConjunction(af::array deviceLHS, af::array deviceRHS) {
    af::array result = af::setIntersect(deviceLHS,deviceRHS,true);
    return result;
}

af::array afCompute::afProduct(af::array deviceLHS, af::array deviceRHS) {
    af::array result = af::operator*(deviceLHS,deviceRHS);
    return result;
}

int afCompute::afSum(af::array deviceData) {
//    af::array device_result = af::sum<int>(data,(dim_t)0);
//    int result = device_result.row(0).elements();
    int result = af::sum<int>(deviceData,(dim_t)0);
    return result;
}

float afCompute::afAvg(af::array deviceData) {
    float sum = af::sum<float>(deviceData,(dim_t)0);
    float result = sum/deviceData.elements();
    return result;
}

int afCompute::afCountIf(af::array deviceData,int value) {
//        return af::count<int>(deviceData);
    af::array index = af::where(af::operator==(deviceData,value));
    return af::count<int>(index);
}

int afCompute::afCount(af::array deviceData) {
    return deviceData.elements();
}


af::array test_nested_loop(af::array::array_proxy result,af::array::array_proxy child,af::array parent) { //

    for(int j=0 ; j < parent.elements(); j++){
        result = af::select(child==parent(j),j,result);
    }
    return result;
}


//af::array nested_loop_join(af::array::array_proxy child,
//                           af::array parent,
//                           af::array::array_proxy result) {
//
//    for (int j = 0; j < parent.elements(); j++){
//        result = af::select(child==parent(j),j,result);
//    }
//    return result;
//}

af::array afCompute::afJoin(af::array parent, af::array child) {

// Why arrayfire join is slow:
// https://stackoverflow.com/questions/50242141/arrayfire-cuda-application-is-extremely-slow-in-the-first-minute
// https://github.com/arrayfire/arrayfire-python/issues/140

/*
 *          condition = (child(i) == parent(j));
            A(i) = (condition)*j + (!condition)*A(i);
 */
    af::array A = af::constant(-1,child.elements());

    gfor(af::seq i, child.elements()) { //
        test_nested_loop(A(i),child(i),parent);
    }
//
    A = A.as(af::dtype::s32);

    return A;
}

// Please note arrayfire does not need prefix sum operation as it already returns the indices in selection operation
// the function has been added below for consistency to run all libraries in sequence
af::array afCompute::afPrefixSum(af::array deviceSelData) {
    return deviceSelData;
}

int afCompute::afFindMax(af::array deviceData) {
    return af::max<int>(deviceData);
}

int afCompute::afFindMin(af::array deviceData) {
    return af::min<int>(deviceData);
}

af::array afCompute::afSort(af::array deviceData, int order) {

    af::array sorted_data;
    if(order){
        sorted_data = af::sort(deviceData,0, false);
    }else{
        sorted_data = af::sort(deviceData);
    }
    return sorted_data;
}

af::array afCompute::afGroupBy(af::array keys, af::array values) {
    af::array keys_out;
    af::array values_out;

    af::sumByKey(keys_out,values_out,keys,values);
    return values_out;
}

af::array afCompute::afCountByKey(af::array data){
    af::array keys_out;
    af::array values_out;

    af::array temp = data;

    af::sumByKey(keys_out,values_out,data,temp);
    values_out = af::operator/(values_out,keys_out);

    return values_out;
}

af::array afCompute::afSumOfVectors(af::array vec1, af::array vec2) {
    af::array result(vec1.elements());

    gfor(af::seq i, vec1.elements()){
        result(i) = vec1(i) + vec2(i);
    }

    result = result.as(af::dtype::s32);

    return result;
}

af::array afCompute::afSortByKey(af::array data, af::array dependent_data, int order) {

    af::array sorted_index;
    af::array sorted_value;
//
//    af::sort(sorted_value,sorted_index,data);
//    af::array result(dependent_data.elements());
//
//
//    gfor(af::seq i,dependent_data.elements()){
//        af::array pos = sorted_index(i);
//        result(i) = dependent_data(pos);
//    }
//
//    result = result.as(s32);
//
//    return result;
    if (order) {
        af::sort(sorted_index, sorted_value, data, dependent_data,0,false);
    }else{
        af::sort(sorted_index, sorted_value, data, dependent_data);
    }
    return sorted_value;
}

af::array afCompute::afAvgByKey(af::array keys, af::array values) {

    af::array sums = afGroupBy(keys,values);

    af::array counts = afCountByKey(keys);

    af::array average = af::operator/(sums,counts);

    return average;
}