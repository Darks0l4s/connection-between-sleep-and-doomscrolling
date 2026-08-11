import matplotlib.pyplot as plt

def close():
    plt.close()

def user_analytics_graph(df):
    fig = plt.Figure(figsize=(20,15))
    
    ax1= fig.add_subplot(2,2, 1)
    ax1.set_title('Gender of People')
    ax1.pie(df['gender'].value_counts(), labels=df['gender'].unique(), autopct='%1.1f%%')
    ax1.legend(loc='upper left', bbox_to_anchor=(-0.3, 1))

    ax2 = fig.add_subplot(2,2, (2,4))
    ax2.set_title('People age')
    ax2.bar(x=df['age'].unique(), height=df['age'].value_counts())
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Count people')

    ax3 = fig.add_subplot(2,2, 3)
    ax3.set_title('People live country')
    ax3.barh(y=df['country_region'].unique(), width=df['country_region'].value_counts())
    ax3.set_xlabel('People count')
    ax3.set_ylabel('Country')
    fig.tight_layout()
    # plt.show()
    # plt.clf()
    # close()
    return fig