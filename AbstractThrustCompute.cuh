//
// Created by hkumar on 21.08.20.
//

#ifndef CROSS_LIBRARY_EXECUTION_ABSTRACTTHRUSTCOMPUTE_CUH
#define CROSS_LIBRARY_EXECUTION_ABSTRACTTHRUSTCOMPUTE_CUH

#endif //CROSS_LIBRARY_EXECUTION_ABSTRACTTHRUSTCOMPUTE_CUH

#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <vector>
#include <string>

using namespace std;

class AbstractThrustCompute{
public:
    thrust::device_vector<int> virtual getThrustGpuData(vector<int> data){}

    void virtual thrustSelection(vector<int> data, string operation, int value, int new_value){}

    void virtual thrustSetUnion(vector<int> lhs, vector<int> rhs){}

    void virtual thrustSetIntersection(vector<int> lhs, vector<int> rhs){}

    void virtual thrustSetDifference(vector<int> lhs, vector<int> rhs){}

    void virtual thrustSort(vector<int> data,int order){}

    void virtual thrustSum(vector<int> data){}

    void virtual thrustFindMax(vector<int> data){}
};

