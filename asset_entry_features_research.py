"""Causal asset-local features evaluated at the next-bar entry timestamp."""
import numpy as np
import pandas as pd
import hashlib
import pickle

FIELDS=("asset_trend_atr","asset_return96_atr","asset_eff96","asset_volume_ratio",
        "asset_close_location","asset_high_distance_atr","asset_low_distance_atr","asset_body_atr")


def feature_table(candle):
    c=pd.DataFrame({k:candle[k] for k in ("time","open","high","low","close","volume")})
    previous=c.close.shift(1)
    tr=pd.concat([c.high-c.low,(c.high-previous).abs(),(c.low-previous).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14,min_periods=14).mean().replace(0,np.nan)
    movement=c.close.diff().abs().rolling(96,min_periods=96).sum().replace(0,np.nan)
    volume=c.volume.shift(1).rolling(20,min_periods=20).median().replace(0,np.nan)
    values=pd.DataFrame(dict(
        asset_trend_atr=(c.close.rolling(20).mean()-c.close.rolling(96).mean())/atr,
        asset_return96_atr=(c.close-c.close.shift(96))/atr,
        asset_eff96=(c.close-c.close.shift(96)).abs()/movement,
        asset_volume_ratio=c.volume/volume,
        asset_close_location=(c.close-c.low)/(c.high-c.low).replace(0,np.nan),
        asset_high_distance_atr=(c.close-c.high.shift(1).rolling(20).max())/atr,
        asset_low_distance_atr=(c.close-c.low.shift(1).rolling(20).min())/atr,
        asset_body_atr=(c.close-c.open)/atr,
    )).shift(1)
    # The signal bar and its 96-bar comparison must precede entry, and all
    # 97 intervening time steps must be contiguous.
    valid=c.time.diff().eq(900).rolling(97,min_periods=97).sum().eq(97)
    values.loc[~valid,:]=np.nan
    values.index=pd.Index(c.time.to_numpy(),name="entry_time")
    return values.replace([np.inf,-np.inf],np.nan)


def attach(frame, paths):
    result=frame.reset_index(drop=True).copy()
    for field in FIELDS:
        result[field]=np.nan
    sources={}
    for (period,symbol),part in result.groupby(['period','symbol'],sort=False):
        path=paths[str(period),str(symbol)]
        raw=path.read_bytes()
        sources[str(path)]=hashlib.sha256(raw).hexdigest()
        table=feature_table(pickle.loads(raw)).reindex(part.entry_time.astype('int64'))
        result.loc[part.index,list(FIELDS)]=table[list(FIELDS)].to_numpy()
    kept=result.dropna(subset=list(FIELDS)).copy()
    kept.attrs['asset_sources']=sources
    kept.attrs['missing_feature_rows']=len(result)-len(kept)
    return kept
