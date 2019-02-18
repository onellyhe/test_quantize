#include "layerinfo.h"

LayerInfo::LayerInfo(std::string name,LayerType type):
    layerType(type),layerName(name)
{
}

LayerInfo::LayerInfo(const LayerInfo &org):
    layerID(org.layerID),layerName(org.layerName),layerType(org.layerType),
    bitWide(org.bitWide),quantity(org.quantity)
{
    setIl(org.inMaxabs,org.outMaxabs,org.paramMaxabs);
}

void LayerInfo::setIl(float inMaxabs,float outMaxabs,float paramMaxabs)
{
    this->inMaxabs = inMaxabs;
    this->outMaxabs = outMaxabs;
    this->paramMaxabs = paramMaxabs;
    inIl = (int)ceil(log2(inMaxabs));
    outIl = (int)ceil(log2(outMaxabs));
    paramIl = (int)ceil(log2(paramMaxabs)+1);
}

std::string LayerInfo::getTypeName()
{
    std::string result = "";
    switch(layerType){
        case CONVOLUTION:result = "CONVOLUTION";break;
        case CONV_RISTRETTO:result = "CONV_RISTRETTO";break;
        case FULLCONNECTION:result = "FULLCONNECTION";break;
        case FC_RISTRETTO:result = "FC_RISTRETTO";break;
        default:result = "CONVOLUTION";
    }
    return result;
}

void LayerInfo::toRistrettoType(){
    switch(layerType){
        case CONVOLUTION:layerType = CONV_RISTRETTO;break;
        case FULLCONNECTION:layerType = FC_RISTRETTO;break;
        default:;
    }
}
