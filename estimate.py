from utils import read_thetas

def hypothesis(thetas, x):
    return thetas[0] + thetas[1] * x

def main():
    try:
        thetas = read_thetas()
        mileage = input("Enter mileage: ")
        mileage = float(mileage)
        estimatePrice = hypothesis(thetas, mileage)
        print("Estimated price: ", estimatePrice)
    except BaseException:
        print("Error during estimation")
        exit()




if __name__ == "__main__":
    main()