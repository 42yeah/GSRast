#pragma once


struct FontData
{
    unsigned int fontSize;
    const unsigned int *fontData;
};

const FontData *getEmbeddedFontData();
