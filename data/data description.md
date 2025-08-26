# Data Description
## Data Provided from Professor
There are 35 commodities future price data in the "raw" folder.  
Each commodity is as follows (all are of contunuous contract 1):  
- BO: US Soybean Oil
- C_: US Corn
- CC: US Cocoa
- CL: US Crude Oil WTI
- CO: London Brent Oil
- CT: US Cotton #2
- DA: US Class III Milk
- FC: US Feeder Cattle
- GC: Gold
- HG: US Copper (per cent)
- HO: US Heating Oil (per cent)
- JN: TOCOM Rubber
- JO: US Orange Juice
- KC: US Coffee C
- LA: London Aluminium
- LB: US Lumber
- LC: US Live Cattle
- LH: US Lean Hogs
- LL: London Lead
- LN: London Nickel
- LT: London Tin
- LX: London Zinc
- NG: US Natural Gas
- O_: US Oats
- PA: US Palladium
- PL: US Platinum
- QS: London Gas Oil
- RR: US Rough Rice
- RS: US Canola
- S_: US Soybeans
- SB: US Sugar #11
- SI: Silver
- SM: US Soybean Meal
- W_: US Wheat
- XB: US Gasoline RBOB (per cent)

Note that in LB dataset, the prices are constant since 2023-05-31, which is incorrect.

## Further Dataset
To build and test my strategies of simple long-short, we need the data of second continuous monthly data for each commodity, which were downloaded from investing.com.  
Notice that for some commodities (shown below) I couldn't get this kind of data, so we will ignore them when constructing protfolio:  
- JN: TOCOM Rubber
- LB: US Lumber
- LL: London Lead
- LN: London Nickel
- LT: London Tin
- LX: London Zinc

## Conclusion Note
Based on the above finding, for rolling window testing the prediction power of shrinkage regression, all dataset except LB would be used, while for long-short strategies backtesting, 5 datasets would be further excluded, ie JN, LL, LN, LT, LX.