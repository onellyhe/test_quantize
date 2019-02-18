#ifndef LAYERINFO_H
#define LAYERINFO_H
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <iostream>

class LayerInfo
{
public:
    enum LayerType{
        CONVOLUTION,
        CONV_RISTRETTO,
        FULLCONNECTION,
        FC_RISTRETTO
    };
    explicit LayerInfo(std::string name,LayerType tyep);
    LayerInfo(const LayerInfo &org);
    void setIl(float inMaxabs,float outMaxabs,float paramMaxabs);
    std::string getTypeName();
    void toRistrettoType();

    std::string layerName;
    LayerType layerType;
    float inMaxabs,outMaxabs,paramMaxabs;
    int inIl,outIl,paramIl;
    int bitWide = 16;
    int layerID;
    long quantity;
};

#endif // LAYERINFO_H
