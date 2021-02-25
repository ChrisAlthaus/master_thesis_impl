import matplotlib.pyplot as plt

#Generate piecharts for detailed evaluation (for better visualizations)

labels = ['Nose', 'Eyes', 'Ears', 'Should.', 'Elbows', 'Wrists', 'Hips', 'Knees', 'Ankles']
fs = 17#14
fig1, ax1 = plt.subplots(figsize=(12, 12))
theme = plt.get_cmap('GnBu')
ax1.set_prop_cycle("color", [theme(1. * i / len(labels))
                             for i in range(len(labels))])
plt.gca().axis("equal")
missdata = [7.2, 10.5, 7.4, 9.1, 14.3, 18.5, 11.6, 11.6, 9.7]
patches, texts, _ = ax1.pie(missdata, autopct='%.1f%%', pctdistance=0.7, textprops={'fontsize': fs})
plt.title('Miss', fontsize=17)
plt.legend(patches,labels, bbox_to_anchor=(1,0.5), loc="center right",
                          bbox_transform=plt.gcf().transFigure, ncol=2, fontsize=14)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
fig1.savefig("piechart-miss.svg")

plt.clf()

fig1, ax1 = plt.subplots(figsize=(12, 12))
theme = plt.get_cmap('GnBu')
ax1.set_prop_cycle("color", [theme(1. * i / len(labels))
                             for i in range(len(labels))])
plt.gca().axis("equal")
inversiondata = [0.0, 4.1, 1.6, 14.9, 7.1, 11.3, 22.1, 18.8, 20.1]
patches, texts, _ = ax1.pie(inversiondata, autopct='%.1f%%', pctdistance=0.7, textprops={'fontsize': fs})
plt.title('Inversion', fontsize=17)
plt.legend(patches,labels, bbox_to_anchor=(1,0.5), loc="center right",
                          bbox_transform=plt.gcf().transFigure, ncol=2, fontsize=14)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
fig1.savefig("piechart-inversion.svg")

plt.clf()


fig1, ax1 = plt.subplots(figsize=(12, 12))
theme = plt.get_cmap('GnBu')
ax1.set_prop_cycle("color", [theme(1. * i / len(labels))
                             for i in range(len(labels))])
plt.gca().axis("equal")
jitterdata = [9.6, 14.5, 12.4, 12.2, 12.3, 10.8, 15.5, 6.5, 6.1]
patches, texts, _ = ax1.pie(jitterdata, autopct='%.1f%%', pctdistance=0.7, textprops={'fontsize': fs})
plt.title('Jitter', fontsize=16)
plt.legend(patches,labels, bbox_to_anchor=(1,0.5), loc="center right",
                          bbox_transform=plt.gcf().transFigure, ncol=2, fontsize=14)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
fig1.savefig("piechart-jitter.svg")

plt.clf()


fig1, ax1 = plt.subplots(figsize=(12, 12))
theme = plt.get_cmap('GnBu')
ax1.set_prop_cycle("color", [theme(1. * i / len(labels))
                             for i in range(len(labels))])
plt.gca().axis("equal")
swapdata = [4.4, 7.5, 4.6, 21.8, 16.5, 14.1, 13.2, 9.4, 8.6]
patches, texts, _ = ax1.pie(swapdata, autopct='%.1f%%', pctdistance=0.7, textprops={'fontsize': fs})
plt.title('Swap', fontsize=16)
plt.legend(patches,labels, bbox_to_anchor=(1,0.5), loc="center right",
                          bbox_transform=plt.gcf().transFigure, ncol=2, fontsize=14)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
fig1.savefig("piechart-swap.svg")

plt.clf()



