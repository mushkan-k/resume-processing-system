import urllib.request, json

# Test year-summary
url = "http://localhost:8080/api/predictions/year-summary"
r = urllib.request.urlopen(url)
d = json.loads(r.read())

print("=== YEAR SUMMARY ===")
print(f"actualDemand (H1): {d['actualDemand']}")
print(f"predictedDemand (H2): {d['predictedDemand']}")
print(f"totalYear: {d['totalYear']}")
print(f"accuracy: {d.get('accuracy')}")
print(f"reliableClusters: {d.get('reliableClusters')}")

print(f"\nbyQuarter: {d['byQuarter']}")

# Test quarterly breakdown
url2 = "http://localhost:8080/api/predictions/quarterly-clusters?quarter=Q1%202026"
try:
    r2 = urllib.request.urlopen(url2)
    d2 = json.loads(r2.read())
    print(f"\n=== Q1 2026 Clusters ===")
    print(f"Total clusters: {d2.get('total', len(d2.get('clusters',[])))}")
    if 'clusters' in d2:
        total_demand = sum(c.get('totalDemand', 0) for c in d2['clusters'])
        print(f"Total demand: {total_demand}")
        print(f"Top 5 clusters:")
        for c in d2['clusters'][:5]:
            print(f"  {c['clusterName']}: {c['totalDemand']}")
except Exception as e:
    print(f"quarterly-clusters error: {e}")

# Try the endpoint the frontend actually uses
url3 = "http://localhost:8080/api/predictions/clusters?quarter=Q1%202026"
try:
    r3 = urllib.request.urlopen(url3)
    d3 = json.loads(r3.read())
    print(f"\n=== /clusters?quarter=Q1 2026 ===")
    print(f"Keys: {list(d3.keys())[:5]}")
    if 'clusters' in d3:
        print(f"Clusters: {d3['total']}")
        total_demand = sum(c.get('totalDemand', 0) for c in d3['clusters'])
        print(f"Total demand: {total_demand}")
except Exception as e:
    print(f"/clusters error: {e}")
