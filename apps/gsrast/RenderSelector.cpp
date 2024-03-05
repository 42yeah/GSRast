#include "RenderSelector.hpp"
#include "DrawBase.hpp"


RenderSelector::RenderSelector() :DrawBase("RenderSelector")
{
    _drawPtr = nullptr;
}

RenderSelector::~RenderSelector()
{

}

void RenderSelector::draw()
{
    if (!_drawPtr) { return; }
    _drawPtr->draw();
}

void RenderSelector::reset(DrawBase::Ptr drawPtr)
{
    _drawPtr = drawPtr;
}
