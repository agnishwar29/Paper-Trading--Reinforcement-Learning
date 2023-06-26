import matplotlib.pyplot as plt
import threading

# Define a lock to synchronize access to the plot
plot_lock = threading.Lock()


def plot_data():
    # Acquire the lock before accessing/modifying the plot
    plot_lock.acquire()

    # Perform plotting operations
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Threaded Plot')

    # Display the plot
    plt.show()

    # Release the lock after finishing the plot
    plot_lock.release()


# Create a thread to run the plot_data function
thread = threading.Thread(target=plot_data)

# Start the thread
thread.start()
