# Tests

## Running a single test module:
To run a single test module, in this case test_custsegm.py:

```sh
$ cd app
$ python -m unittest tests.test_custsegm
Just reference the test module the same way you import it.
```

## Running a single test case or test method:
Also you can run a single TestCase or a single test method:

```sh
$ cd app
$ python -m unittest test.test_custsegm.CustomerSegmentationTestCase
$ python -m unittest test.test_custsegm.CustomerSegmentationTestCase.test_method
```

## Running all tests:
This will run all the test*.py modules inside the test package.

```sh
$ cd app
$ python -m unittest
```

You can also use test discovery which will discover and run all the tests for you, they must be modules or packages named test*.py (can be changed with the -p, --pattern flag):

```sh
$ cd app
$ python -m unittest discover
```

See: https://stackoverflow.com/a/24266885






