```
   [mutating]
    bool addNeeVertex(RestirPathTracerParams params, uint pathLength, float3 wi, float3 pathWeight, float3 postfixWeight, bool useHybridShift,
        float russianroulettePDF, float misWeight, float lightPdf, int lightType, inout PathReservoir pathReservoir, bool forceAdd = false)

```

```
if (forceAdd || pathReservoir.add(pathWeight, russianroulettePDF, sg))
```

最初の重みは、w = pathWeight(Lr)/rusianroulettePDFで計算
```
        float w = toScalar(in_F) / p;

        if (isnan(w) || w == 0.f) return false;

        weight += w;
```
つまり、mi(xi)p_hat(xi)Wi = Lr/rrPDF
mi(xi)=1であり、p_hat(i)がこいつ↓
Lrは、Lr = BSDFCosineWeight() * ls.Li * throughput
となるとWiがrrPDFということになる


```
    bool terminatePathByRussianRoulette(inout PathState path)
    {
        const float rrVal = luminance(path.getCurrentThp());
        const float prob = max(0.f, 1.f - rrVal);
        if (sampleNext1D(path.sg) < prob)
        {
            path.terminate();
            return true;
        }
        // 生き残る確率が乗算されていく
        path.russianRoulettePdf *= 1.f - prob;

        return false;
    }
```

つまり！！！！！長さKのレイが！！！！！！！！生き残る確率が！！！！！！！！！！！！InitialResamplingのぉぉぉ！！！！！！！！！！！！！！！！！！！！！！！！PDFに！！！！！！！！！！！！！！！！！！！なっていますううううううう！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！