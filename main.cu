#pragma once
#include "evaluation/executeQueries.h"
#include <iostream>


int main(){
    auto queries = new executeTpchQueries();
    queries->call_tpch_query(6);
    queries->transfer_time(8);
}


