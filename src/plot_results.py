#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path="results.csv"):
    data={}
    with open(path) as f:
        reader=csv.DictReader(f)
        for c in reader.fieldnames: data[c]=[]
        for row in reader:
            for c in reader.fieldnames: data[c].append(float(row[c]))
    return [int(x) for x in data["nodes"]], data

def plot_metric(nodes, beb, ql, title, ylabel, filename):
    plt.figure(figsize=(7,5))
    plt.plot(nodes,beb,"o--",color="#c0392b",linewidth=2,markersize=8,label="BEB (IEEE 802.15.4)")
    plt.plot(nodes,ql,"s-",color="#1f6feb",linewidth=2,markersize=8,label="QL-CW (proposed)")
    plt.title(title,fontsize=13,fontweight="bold")
    plt.xlabel("Number of Nodes (N)",fontsize=11); plt.ylabel(ylabel,fontsize=11)
    plt.grid(True,linestyle=":",alpha=0.6); plt.legend(fontsize=11); plt.xticks(nodes)
    plt.tight_layout(); plt.savefig(filename,dpi=200); plt.close()
    print(f"[OK] saved {filename}")

def main():
    nodes,d=load()
    plot_metric(nodes,d["beb_collisions"],d["ql_collisions"],
                "Collisions vs. Number of Nodes","Total Collisions","fig_collisions.png")
    plot_metric(nodes,d["beb_pdr"],d["ql_pdr"],
                "PDR vs. Number of Nodes","Packet Delivery Ratio (%)","fig_pdr.png")
    plot_metric(nodes,d["beb_delay"],d["ql_delay"],
                "Average Delay vs. Number of Nodes","Average End-to-End Delay (s)","fig_delay.png")
    plot_metric(nodes,d["beb_energy"],d["ql_energy"],
                "Energy per Node vs. Number of Nodes","Average Energy per Node (units)","fig_energy.png")

if __name__=="__main__":
    main()
