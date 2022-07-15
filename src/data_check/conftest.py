import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")
    parser.addoption("--min_longitude", action="store")
    parser.addoption("--max_longitude", action="store")
    parser.addoption("--min_latitude", action="store")
    parser.addoption("--max_latitude", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)

@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)

@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)


@pytest.fixture(scope='session')
def min_longitude(request):
    min_longitude = request.config.option.min_longitude

    if min_longitude is None:
        pytest.fail("You must provide min_longitude")

    return float(min_longitude)


@pytest.fixture(scope='session')
def max_longitude(request):
    max_longitude = request.config.option.max_longitude

    if max_longitude is None:
        pytest.fail("You must provide max_longitude")

    return float(max_longitude)


@pytest.fixture(scope='session')
def min_latitude(request):
    min_latitude = request.config.option.min_latitude

    if min_latitude is None:
        pytest.fail("You must provide min_latitude")

    return float(min_latitude)


@pytest.fixture(scope='session')
def max_latitude(request):
    max_latitude = request.config.option.max_latitude

    if max_latitude is None:
        pytest.fail("You must provide max_latitude")

    return float(max_latitude)