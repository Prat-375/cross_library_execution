//
// Created by hkumar on 26.08.20.
//

#include "ThrustComputeOps.cuh"

thrust::device_vector<int> ThrustCompute::getThrustGpuData(vector<int> data) {
    thrust::host_vector<int> hostData(data);
    thrust::device_vector<int> deviceData = hostData;

    return deviceData;
}

vector<int> ThrustCompute::getThrustCpuData(thrust::device_vector<int> d_data) {
    vector<int> h_data(d_data.size());
    thrust::copy(d_data.begin(),d_data.end(),h_data.begin());
    return h_data;
}
vector<float> ThrustCompute::getThrustCpuData(thrust::device_vector<float> d_data) {
    vector<float> h_data(d_data.size());
    thrust::copy(d_data.begin(),d_data.end(),h_data.begin());
    return h_data;
}

thrust::device_vector<int> ThrustCompute::thrustIN(thrust::device_vector<int> dataToFilter,
                                                   thrust::device_vector<int> refData,
                                                   thrust::device_vector<int> prefixSum){

    const int *m_data = thrust::raw_pointer_cast(dataToFilter.data());
    const int *m_ref = thrust::raw_pointer_cast(refData.data());
    const int *m_ps = thrust::raw_pointer_cast(prefixSum.data());
    std::size_t ps_size = prefixSum.size();

    thrust::device_vector<int> deviceResult(dataToFilter.size());
    int *m_result = thrust::raw_pointer_cast(deviceResult.data());

//    thrust::transform(dataToFilter.begin(),dataToFilter.end(),deviceResult.begin(),[=] __device__ (const int x){
//        for(int i = 0; i < ps_size; i++){
//            if(x == m_ref[i]){
//                return 1;
//            }
//        }
//    });

    thrust::for_each_n(thrust::counting_iterator<int>(0),dataToFilter.size(),[=] __device__ (const std::size_t x){
        for(int i = 0; i < ps_size; i++){
            if(m_data[x] == m_ref[m_ps[i]]){
                m_result[x] = 1;
            }
        }
    });
    return deviceResult;
}

thrust::device_vector<int> ThrustCompute::thrustSelection(thrust::device_vector<int> deviceData, string operation, int value){

    thrust::device_vector<int> device_result(deviceData.size());

    if(!operation.compare("GE")) {

        thrust::transform(thrust::device,deviceData.begin(),deviceData.end(),device_result.begin(),[=] __device__(const int x) {
            return (x >= value);
        });
    }
    else if(!operation.compare("LE")) {

        thrust::transform(thrust::device,deviceData.begin(),deviceData.end(),device_result.begin(),[=] __device__(const int x) {
            return (x <= value);
        });
    }
    else if(!operation.compare("G")) {
        thrust::transform(thrust::device,deviceData.begin(),deviceData.end(),device_result.begin(),[=] __device__(const int x) {
            return (x > value);
        });
    }
    else if(!operation.compare("L")) {
        thrust::transform(thrust::device,deviceData.begin(),deviceData.end(),device_result.begin(),[=] __device__(const int x) {
            return (x < value);
        });
    }
    else if(!operation.compare("EQ")) {
        thrust::transform(thrust::device,deviceData.begin(),deviceData.end(),device_result.begin(),[=] __device__(const int x) {
            return (x == value);
        });
    }
    else{
        thrust::transform(thrust::device,deviceData.begin(),deviceData.end(),device_result.begin(),[=] __device__(const int x) {
            return (x != value);
        });
    }

    return device_result;
}

thrust::device_vector<int> ThrustCompute::thrustSelectionArrays(thrust::device_vector<int> deviceLHS, string operation,
                                                                thrust::device_vector<int> deviceRHS) {

    thrust::device_vector<int> result_dev(deviceLHS.size());


    size_t v1size = deviceLHS.size();
    const int *m_vec1 = thrust::raw_pointer_cast(deviceLHS.data());
    const int *m_vec2 = thrust::raw_pointer_cast(deviceRHS.data());
    int *m_result = thrust::raw_pointer_cast(result_dev.data());


    if(!operation.compare("GE")) {

        thrust::for_each_n( thrust::device,
                            thrust::counting_iterator<size_t>(0),
                            (v1size),
                            [=] __device__ (const std::size_t x){
                                m_result[x] = (m_vec1[x] >= m_vec2[x]);
                            });

    }

    else if(!operation.compare("LE")) {

        thrust::for_each_n( thrust::device,
                            thrust::counting_iterator<size_t>(0),
                            (v1size),
                            [=] __device__ (const std::size_t x){
                                m_result[x] = (m_vec1[x] <= m_vec2[x]);
                            });

    }

    else if(!operation.compare("G")) {

        thrust::for_each_n( thrust::device,
                            thrust::counting_iterator<size_t>(0),
                            (v1size),
                            [=] __device__ (const std::size_t x){
                                m_result[x] = (m_vec1[x] > m_vec2[x]);
                            });

    }

    else if(!operation.compare("L")) {

        thrust::for_each_n( thrust::device,
                            thrust::counting_iterator<size_t>(0),
                            (v1size),
                            [=] __device__ (const std::size_t x){
                                m_result[x] = (m_vec1[x] < m_vec2[x]);
                            });

    }

    else if(!operation.compare("EQ")) {

        thrust::for_each_n( thrust::device,
                            thrust::counting_iterator<size_t>(0),
                            (v1size),
                            [=] __device__ (const std::size_t x){
                                m_result[x] = (m_vec1[x] == m_vec2[x]);
                            });

    }

    else{

        thrust::for_each_n( thrust::device,
                            thrust::counting_iterator<size_t>(0),
                            (v1size),
                            [=] __device__ (const std::size_t x){
                                m_result[x] = (m_vec1[x] != m_vec2[x]);
                            });

    }

    return result_dev;
}

thrust::device_vector<int> ThrustCompute::thrustConjunction(thrust::device_vector<int> deviceLHS,thrust::device_vector<int> deviceRHS) {

    thrust::device_vector<int> device_result(deviceLHS.size());

    thrust::transform(deviceLHS.begin(),deviceLHS.end(),deviceRHS.begin(),device_result.begin(),thrust::bit_and<int>());

    return device_result;
}

thrust::device_vector<int> ThrustCompute::thrustSort(thrust::device_vector<int> deviceData, int order){
    if(order){
        thrust::sort(thrust::device, deviceData.begin(),deviceData.end(),thrust::greater<int>());
    }else{
        thrust::sort(deviceData.begin(),deviceData.end());
    }
    return deviceData;
}

thrust::device_vector<int> ThrustCompute::thrustSortByKey(thrust::device_vector<int> deviceData,
                                                          thrust::device_vector<int> dependentData,
                                                          int order) {

    if(order) {
        thrust::sort_by_key(deviceData.begin(), deviceData.end(), dependentData.begin(), thrust::greater<int>());
    }else{
        thrust::sort_by_key(deviceData.begin(), deviceData.end(), dependentData.begin());
    }
//    thrust::device_vector<std::size_t> index (deviceData.size());
//    thrust::sequence(index.begin(), index.end(), 0);
//
//    const int *m_vec1 = thrust::raw_pointer_cast(deviceData.data());
//
//    if (order){
//        thrust::sort(index.begin(), index.end(), [=] __device__ (size_t a, size_t b) { return m_vec1[a] > m_vec1[b]; });
//    }else{
//        thrust::sort(index.begin(), index.end(), [=] __device__ (size_t a, size_t b) { return m_vec1[a] < m_vec1[b]; });
//    }
//
//    thrust::device_vector<int> deviceResult(dependentData.size());
//    thrust::gather(index.begin(),index.end(),dependentData.begin(),deviceResult.begin());
//    return deviceResult;
    return dependentData;
}

thrust::device_vector<int> ThrustCompute::thrustProduct(thrust::device_vector<int> deviceLHS, thrust::device_vector<int> deviceRHS){
    thrust::device_vector<int> device_result(deviceLHS.size());

    thrust::transform(deviceLHS.begin(),deviceLHS.end(),deviceRHS.begin(),device_result.begin(),thrust::multiplies<int>());

    return device_result;
}

int ThrustCompute::thrustSum(thrust::device_vector<int> deviceData) {
    int result = thrust::reduce(deviceData.begin(),deviceData.end(),0);
    return result;
}

float ThrustCompute::thrustAvg(thrust::device_vector<int> deviceData) {

    int total = thrust::reduce(deviceData.begin(),deviceData.end(),0);
    float result = (float)total/deviceData.size();
    return result;
}

int ThrustCompute::thrustCountIf(thrust::device_vector<int> deviceData,int value) {
    // counting number of 1s is enough here
    return thrust::count(deviceData.begin(),deviceData.end(),value);
}

int ThrustCompute::thrustCount(thrust::device_vector<int> deviceData) {
    return deviceData.size();
}

thrust::device_vector<int> ThrustCompute::thrustJoin(thrust::device_vector<int> vec1_dev, thrust::device_vector<int> vec2_dev) {

    thrust::device_vector<int> result_find(vec2_dev.size());

    size_t v1size = vec1_dev.size();
    const int *m_vec1 = thrust::raw_pointer_cast(vec1_dev.data());
    const int *m_vec2 = thrust::raw_pointer_cast(vec2_dev.data());
    int *m_result = thrust::raw_pointer_cast(result_find.data());


    thrust::for_each_n(thrust::counting_iterator<size_t>(0),vec2_dev.size(),[=] __device__ (const int x){
        m_result[x] = -1;
        for(int i=0; i < v1size; i++){
            if (m_vec2[x] == m_vec1[i]){
                m_result[x] = i;
            }
        }
    });


    return result_find;
}

thrust::device_vector<int> ThrustCompute::thrustPrefixSum(thrust::device_vector<int> deviceSelData) {

    thrust::device_vector<int> dataIndex(deviceSelData.size());
    thrust::sequence(dataIndex.begin(),dataIndex.end(),0);

    thrust::device_vector<int> ps(deviceSelData.size());
    thrust::exclusive_scan(deviceSelData.begin(),deviceSelData.end(),ps.begin(),0); // prefix sum result

    int size = thrust::count(deviceSelData.begin(),deviceSelData.end(),1);
    thrust::device_vector<int> device_result(size);

//    for(int i = 0; i < deviceData.size(); i++){
//        if(deviceData[i]){
//            device_result[ps[i]] = i;
//        }
//    }

    thrust::scatter_if(dataIndex.begin(),dataIndex.end(),ps.begin(),deviceSelData.begin(),device_result.begin());

    return device_result;
}

thrust::device_vector<int> ThrustCompute::thrustPrefixSum(thrust::device_vector<int> deviceSelData,
                                                          thrust::device_vector<int> data) {

    thrust::device_vector<int> ps(deviceSelData.size());
    thrust::exclusive_scan(deviceSelData.begin(),deviceSelData.end(),ps.begin()); // prefix sum result
//
    int size = ps.back();
    thrust::device_vector<int> device_result(size);

    thrust::scatter_if(data.begin(),data.end(),ps.begin(),deviceSelData.begin(),device_result.begin());

    return device_result;
}

int ThrustCompute::thrustFindMax(thrust::device_vector<int> data){
    thrust::device_vector<int>::iterator it = thrust::max_element(data.begin(),data.end());
    int index = thrust::distance(data.begin(),it);
    return data[index];
}

int ThrustCompute::thrustFindMin(thrust::device_vector<int> data){
//    return thrust::reduce(thrust::device,data.begin(),data.end(),-1,thrust::maximum<int>());
    thrust::device_vector<int>::iterator it = thrust::min_element(data.begin(),data.end());
    int index = thrust::distance(data.begin(),it);
    return data[index];
}

thrust::device_vector<int> ThrustCompute::thrustGroupBy(thrust::device_vector<int> keys,
                                                   thrust::device_vector<int> values) {

    thrust::device_vector<int> temp = keys;
    auto last = thrust::unique(temp.begin(),temp.end());
    temp.erase(last, temp.end());
    int size = temp.size();

    thrust::device_vector<int> oKeys(size); //previous size here: keys.size()
    thrust::device_vector<int> oVals(size); //previous size here: values.size()

    thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), oKeys.begin(), oVals.begin());
    return oVals;
}

thrust::device_vector<int> ThrustCompute::thrustHashJoin(thrust::device_vector<int> parent, thrust::device_vector<int> child) {

    thrust::device_vector<int> hashParent(parent.size());   // build hash table
    thrust::device_vector<int> moduloParent(parent.size()); // hash function

    thrust::fill(moduloParent.begin(), moduloParent.end(), parent.size());

    // build table done - works only for unique column values
    thrust::transform(parent.begin(),parent.end(),moduloParent.begin(),hashParent.begin(),thrust::modulus<int>());

    thrust::device_vector<int> moduloChild(child.size());
    thrust::fill(moduloChild.begin(),moduloChild.end(),parent.size());

    thrust::device_vector<int> hashChild(child.size());

    //probe table
    thrust::transform(child.begin(),child.end(),moduloChild.begin(),hashChild.begin(),thrust::modulus<int>());

    return hashChild;
}

thrust::device_vector<int> ThrustCompute::thrustCountByKey(thrust::device_vector<int> data) {
    thrust::device_vector<int> temp = data;
    auto last = thrust::unique(temp.begin(),temp.end());
    temp.erase(last, temp.end());
    int size = temp.size();

    thrust::device_vector<int> C(size); //previous size here: data.size()
    thrust::device_vector<int> D(size); //previous size here: data.size()

    thrust::reduce_by_key(data.begin(),data.end(),thrust::make_constant_iterator(1),C.begin(),D.begin(),thrust::equal_to<int>(),thrust::plus<int>());

    return D;
}

struct sum_and_count
{
    template <typename Tuple>
    __host__ __device__
    Tuple operator()(const Tuple& a, const Tuple& b) const
    {
        return Tuple(thrust::get<0>(a) + thrust::get<0>(b),
                     thrust::get<1>(a) + thrust::get<1>(b));
    }
};

thrust::device_vector<float> ThrustCompute::thrustAvgByKey(thrust::device_vector<int> k,
                                                         thrust::device_vector<int> v) {

    thrust::device_vector<int> temp = k;

    auto last = thrust::unique(temp.begin(),temp.end());
    temp.erase(last, temp.end());

    int size = temp.size();

    thrust::device_vector<int> keys(size);
    thrust::device_vector<int> sums(size);
    thrust::device_vector<int> counts(size);
    thrust::device_vector<float> result(size);

    thrust::reduce_by_key
            (k.begin(), k.end(),
             thrust::make_zip_iterator(thrust::make_tuple(v.begin(), thrust::constant_iterator<int>(1))),
             keys.begin(),
             thrust::make_zip_iterator(thrust::make_tuple(sums.begin(), counts.begin())),
             thrust::equal_to<int>(),
             sum_and_count());

    vector<int> h_sums = getThrustCpuData(sums);
    vector<int> h_count = getThrustCpuData(counts);

    thrust::transform(sums.begin(), sums.end(), counts.begin(), result.begin(),
                      thrust::divides<float>());

    return result;
}

thrust::device_vector<int> ThrustCompute::thrustGather(thrust::device_vector<int> index,
                                                       thrust::device_vector<int> values) {

    thrust::device_vector<int> deviceResult(index.size());

    thrust::gather(index.begin(),index.end(),values.begin(),deviceResult.begin());
    return deviceResult;
}

thrust::device_vector<int> ThrustCompute::thrustSumOfVectors(thrust::device_vector<int> array1,
                                                             thrust::device_vector<int> array2) {

    thrust::device_vector<int> deviceResult(array1.size());

    const int *m_vec1 = thrust::raw_pointer_cast(array1.data());
    const int *m_vec2 = thrust::raw_pointer_cast(array2.data());

    int *m_result = thrust::raw_pointer_cast(deviceResult.data());

    thrust::for_each_n(thrust::counting_iterator<int>(0),array1.size(),[=] __device__ (const std::size_t x){
                m_result[x] = m_vec1[x] + m_vec2[x];
    });

    return deviceResult;
}