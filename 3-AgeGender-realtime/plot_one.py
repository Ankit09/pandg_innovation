import matplotlib.pyplot as plt
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
fracs = [15, 30, 45, 10]

fig = plt.figure()

ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)
ax1.pie(fracs, labels=labels, radius = 1.2)
ax2 = fig.add_axes([.5, .0, .5, .5], aspect=1)
ax2.pie(fracs, labels=labels, radius = 1.2)
ax1.set_title('Title for ax1')
ax2.set_title('Title for ax2')
plt.show()