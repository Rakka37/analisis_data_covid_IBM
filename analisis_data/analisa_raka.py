#NAMA             : Ramdhani Rakka Levine Nathaniel
#NAMA UNIVERSITAS : Universitas 17 Agustus 1945 Surabaya
#DOMISILI         : Bojonegoro

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#styling dasar
sns.set(style="whitegrid", font_scale=1.05)
pd.options.display.float_format = '{:,.0f}'.format

#baca file CSV nya guys
FILE = "Indonesia_coronavirus_daily_data.csv"
if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

df = pd.read_csv(FILE, low_memory=False)

#untuk merapikan nama kolom
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

#pastikan kolom numerik dibaca sebagai angka ya guyss
cols = ["daily_case","daily_death","daily_recovered",
        "active_case","cumulative_case","cumulative_recovered",
        "cumulative_death","cumulative_active_case"]
for c in cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

#data nasional 'time series'
if "province" in df.columns:
    nat = df.groupby("date")[["daily_case","daily_death","daily_recovered"]].sum().reset_index()
else:
    nat = df[["date","daily_case","daily_death","daily_recovered"]].copy()

nat = nat.sort_values("date").reset_index(drop=True)
nat["cumulative_cases"] = nat["daily_case"].cumsum()
nat["cumulative_deaths"] = nat["daily_death"].cumsum()
nat["cumulative_recovered"] = nat["daily_recovered"].cumsum()
nat["cumulative_active"] = nat["cumulative_cases"] - nat["cumulative_recovered"] - nat["cumulative_deaths"]

#statistik nasional
total_case = int(nat["cumulative_cases"].iloc[-1])
total_death = int(nat["cumulative_deaths"].iloc[-1])
total_recovered = int(nat["cumulative_recovered"].iloc[-1])
last_date = nat["date"].max().date()

peak_case_row = nat.loc[nat["daily_case"].idxmax()] if nat["daily_case"].notna().any() else None
peak_death_row = nat.loc[nat["daily_death"].idxmax()] if nat["daily_death"].notna().any() else None

#snapshot per provinsi 'data terakhir'
if "province" in df.columns:
    latest_idx = df.groupby("province")["date"].idxmax()
    df_last = df.loc[latest_idx].copy()
else:
    df_last = df.copy()

#hitung CFR dan recovery rate
for c in ["cumulative_case","cumulative_death","cumulative_recovered"]:
    if c not in df_last.columns:
        df_last[c] = 0

df_last["cfr"] = np.where(df_last["cumulative_case"]>0,
                          (df_last["cumulative_death"]/df_last["cumulative_case"])*100, np.nan)
df_last["recovery_rate"] = np.where(df_last["cumulative_case"]>0,
                                    (df_last["cumulative_recovered"]/df_last["cumulative_case"])*100, np.nan)

#ranking provinsi
top_case = df_last.sort_values("cumulative_case", ascending=False).head(10)
top_death = df_last.sort_values("cumulative_death", ascending=False).head(10)
bottom_case = df_last.sort_values("cumulative_case", ascending=True).head(10)
bottom_death = df_last.sort_values("cumulative_death", ascending=True).head(10)
top_cfr = df_last.sort_values("cfr", ascending=False).head(10)
low_cfr = df_last.sort_values("cfr", ascending=True).head(10)

#fungsi untuk cetak tbl dg nomor urut
def print_table(df_tbl, cols, title):
    tbl = df_tbl[cols].reset_index(drop=True).reset_index().rename(columns={"index":"No"})
    print(f"\n=== {title} ===")
    print(tbl.to_string(index=False))

#cetak statistik nasional
print(f"\nData terakhir: {last_date}")
print(f"Total kumulatif kasus: {total_case:,}")
print(f"Total kumulatif kematian: {total_death:,}")
print(f"Total kumulatif sembuh: {total_recovered:,}")

if peak_case_row is not None:
    print(f"Puncak kasus harian: {int(peak_case_row['daily_case']):,} pada {pd.to_datetime(peak_case_row['date']).date()}")
if peak_death_row is not None:
    print(f"Puncak kematian harian: {int(peak_death_row['daily_death']):,} pada {pd.to_datetime(peak_death_row['date']).date()}")

#cetak analisis per provinsi
if "province" in df.columns:
    print_table(top_case, ["province","cumulative_case"], "PROVINSI DENGAN KASUS TERTINGGI")
    print_table(top_death, ["province","cumulative_death"], "PROVINSI DENGAN KEMATIAN TERTINGGI")
    print_table(bottom_case, ["province","cumulative_case"], "PROVINSI DENGAN KASUS TERENDAH")
    print_table(bottom_death, ["province","cumulative_death"], "PROVINSI DENGAN KEMATIAN TERENDAH")
    print_table(top_cfr[["province","cfr"]], ["province","cfr"], "PROVINSI DENGAN CFR TERTINGGI")
    print_table(low_cfr[["province","cfr"]], ["province","cfr"], "PROVINSI DENGAN CFR TERENDAH")

#grafik harian nasional
nat["roll_case"] = nat["daily_case"].rolling(7, min_periods=1).mean()
plt.figure(figsize=(12,5))
plt.plot(nat["date"], nat["daily_case"], alpha=0.4, label="Kasus Harian")
plt.plot(nat["date"], nat["roll_case"], linewidth=2, label="Rata-rata 7 hari")
plt.title("Kasus Harian Nasional")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah")
plt.legend()
plt.tight_layout()
plt.show()

#grafik kumulatif nasional
plt.figure(figsize=(12,5))
plt.plot(nat["date"], nat["cumulative_cases"], label="Kasus kumulatif")
plt.plot(nat["date"], nat["cumulative_recovered"], label="Sembuh kumulatif")
plt.plot(nat["date"], nat["cumulative_deaths"], label="Kematian kumulatif")
plt.title("Kumulatif Nasional")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah")
plt.legend()
plt.tight_layout()
plt.show()

#fungsi bikin bar chart hrzntl dg label angka
def hbar_with_labels(df_plot, x, y, title, cmap="Blues"):
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x=x, y=y, data=df_plot, hue=y, dodge=False, legend=False, palette=cmap)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width):,}", xy=(width, p.get_y() + p.get_height()/2),
                    xytext=(5,0), textcoords="offset points", ha="left", va="center", fontsize=9)
    plt.title(title)
    plt.xlabel(x.replace("_"," ").title())
    plt.ylabel(y.replace("_"," ").title())
    plt.tight_layout()
    plt.show()

#grafik per-provinsi
if "province" in df.columns:
    hbar_with_labels(top_case, "cumulative_case", "province", f"Top 10 Provinsi - Kasus (s/d {last_date})", cmap="Blues_r")
    hbar_with_labels(top_death, "cumulative_death", "province", f"Top 10 Provinsi - Kematian (s/d {last_date})", cmap="Reds_r")
    
    #grafik CFR per provinsi 'top 15'
    plt.figure(figsize=(10,6))
    ax = sns.barplot(
        x="province", y="cfr",
        hue="province", dodge=False, legend=False,
        data=df_last.sort_values("cfr", ascending=False).head(15),
        palette="coolwarm"
    )
    plt.xticks(rotation=90)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.title("CFR Provinsi (Top 15)")
    plt.ylabel("CFR (%)")
    plt.tight_layout()
    plt.show()

#simpan snapshot terakhir
OUT = "snapshot_provinsi.csv"
df_last.to_csv(OUT, index=False)
print("\nSnapshot per-provinsi disimpan:", OUT)
