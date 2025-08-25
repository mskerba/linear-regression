from utils import read_data, plot_cost_surface_3d, plotting_data
import math

def cost_function(theta0, theta1, x, y):
    m = len(x)
    total = 0.0
    for i in range(m):
        pred = theta0 + theta1 * x[i]
        total += abs(pred - y[i])
    return total / (m)

def train(mileages, prices):
    # --- GD hyperparams
    theta0, theta1 = 0.0, 0.0
    alpha = 1e-4         # smaller lr works after scaling
    max_epoch = 100000

    m = len(mileages)
    prev_loss = float("inf")

    for epoch in range(max_epoch):
        # gradients (batch)
        err_sum = 0.0
        errx_sum = 0.0
        for i in range(m):
            err = (theta0 + theta1 * mileages[i]) - prices[i]
            err_sum += err
            errx_sum += err * mileages[i]
        # tmp updates (simultaneous)
        tmp0 = theta0 - alpha * (err_sum / m)
        tmp1 = theta1 - alpha * (errx_sum / m)
        theta0, theta1 = tmp0, tmp1
        loss = cost_function(theta0, theta1, mileages, prices)
        if round(prev_loss, 7) == round(
                loss, 7
            ):
            break
        prev_loss = loss
    return theta0, theta1

def thetas_unscale_and_store(theta0, theta1, mx, sx):
    try:
        # Unscale thetas
        theta0 -= theta1 * mx / sx
        theta1 /= sx

        # Store thetas
        output = open("thetas", "w")
        output.write(str(theta0) + "," + str(theta1))
        output.close()
    except BaseException as e:
        print("Error during theta storing: ", e)
        exit()


def program_evaluation(mileages_raw, prices, theta0, theta1):
    mape = (1/len(mileages_raw)) * sum(
    abs((theta0 + theta1 * mileages_raw[i] - prices[i]) / prices[i]) 
    for i in range(len(mileages_raw))
    ) * 100

    print(f"MAPE: {mape}%")

    accuracy_like = 100 - mape
    print(f"Accuracy-like: {accuracy_like}%")


    y_true = prices
    y_pred = [theta0 + theta1 * m for m in mileages_raw]

    # moyenn dial l prix
    y_mean = sum(y_true) / len(y_true)

    # SS_tot = total sum of squares
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)

    # SS_res = residual sum of squares
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))

    # R²
    r2 = 1 - (ss_res / ss_tot)

    print(f"R² score (accuracy-like): {r2 * 100:.2f}%")
    return mape, accuracy_like, r2


def main():
    try:
        # --- load
        file = "data.csv"
        raw = read_data(file)
        mileages_raw = [r[0] for r in raw]
        prices = [r[1] for r in raw]
    except BaseException as e:
        print("Error during data loading: ", e)
        exit()
    
    try:
        # --- feature scaling (standardization)
        mx = sum(mileages_raw)/len(mileages_raw)
        sx = (sum((v - mx)**2 for v in mileages_raw)/len(mileages_raw))**0.5 
        mileages = [(v - mx)/sx for v in mileages_raw]
        theta0,theta1 = train(mileages, prices)
    except BaseException as e:
        print("Error during training: ", e)
        exit()
    thetas_unscale_and_store(theta0, theta1, mx, sx)

    theta0 -= theta1 * mx / sx
    theta1 /= sx

    # --- evaluation
    program_evaluation(mileages_raw, prices, theta0, theta1)

    # --- plotting data
    plotting_data(mileages_raw, prices, theta0, theta1)

    # --- plotting cost surface(3D)
    T0, T1, J = plot_cost_surface_3d(mileages, prices, theta_hat=None)







if __name__ == "__main__":
    main()
