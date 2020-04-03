/* This sample is based on the Chapter 1 from book
 * "Building Machine Learning Systems with Python" by Willi Richert
 */

// third party includes
#include <csv.h>
#include <plot.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xio.hpp>

// stl includes
#include <algorithm>
#include <experimental/filesystem>
#include <iostream>
#include <random>
#include <string>

// application includes
#include "../ioutils.h"
#include "../utils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;
typedef double DType;

// linalg package doesn't support dynamic layouts
using Matrix = xt::xarray<DType, xt::layout_type::row_major>;

auto standardize(const Matrix& v) {
  assert(v.shape().size() == 1);
  auto m = xt::eval(xt::mean(v))[0];
  auto n = v.shape()[0];
  auto sd = xt::eval(
      xt::sqrt(xt::sum(xt::pow(v - m, 2)) / static_cast<DType>(n - 1)))[0];
  auto sv = (v - m) / sd;
  return std::make_tuple(xt::eval(sv), m, sd);
}

auto minmax_scale(const Matrix& v, DType rmin, DType rmax) {
  assert(v.shape().size() == 1);
  auto minmax = xt::minmax(v)();
  Matrix vs = (v - minmax[0]) / (minmax[1] - minmax[0]);
  return xt::eval(vs * (rmax - rmin) - rmin);
}

auto generate_polynomial(const Matrix& x, size_t degree) {
  assert(x.shape().size() == 1);
  auto rows = x.shape()[0];
  auto poly_shape = std::vector<size_t>{rows, degree};
  Matrix poly_x = xt::zeros<DType>(poly_shape);
  // fill additional column for simpler vectorization
  {
    auto xv = xt::view(poly_x, xt::all(), 0);
    xv = xt::ones<DType>({rows});
  }
  // copy initial data
  {
    auto xv = xt::view(poly_x, xt::all(), 1);
    xv = std::get<0>(standardize(x));
  }
  // generate additional terms
  auto x_col = xt::view(poly_x, xt::all(), 1);
  for (size_t i = 2; i < degree; ++i) {
    auto xv = xt::view(poly_x, xt::all(), i);
    xv = xt::pow(x_col, static_cast<DType>(i));
    xv = std::get<0>(standardize(xv));
  }
  return poly_x;
}

auto bgd(const Matrix& x, const Matrix& y, size_t batch_size) {
  size_t n_epochs = 50000;
  DType lr = 0.0055;

  auto rows = x.shape()[0];
  auto cols = x.shape()[1];

  size_t batches = rows / batch_size;  // some samples will be skipped
  Matrix b = xt::zeros<DType>({cols});

  DType prev_cost = std::numeric_limits<DType>::max();
  for (size_t i = 0; i < n_epochs; ++i) {
    for (size_t bi = 0; bi < batches; ++bi) {
      auto s = bi * batch_size;
      auto e = s + batch_size;
      Matrix batch_x = xt::view(x, xt::range(s, e), xt::all());
      Matrix batch_y = xt::view(y, xt::range(s, e), xt::all());

      auto yhat = xt::linalg::dot(batch_x, b);
      Matrix error = yhat - batch_y;

      auto grad = xt::linalg::dot(xt::transpose(batch_x), error) /
                  static_cast<DType>(batch_size);

      b = b - lr * grad;
    }

    auto cost = (xt::sum(xt::pow(y - xt::linalg::dot(x, b), 2.f)) /
                 static_cast<DType>(rows))(0);  // evaluate value immediatly

    std::cout << "Iteration : " << i << " Cost = " << cost << std::endl;
    if (cost <= prev_cost)
      prev_cost = cost;
    else
      break;  // early stopping
  }
  return b;
}

auto make_regression_model(const Matrix& data_x,
                           const Matrix& data_y,
                           size_t p_degree,
                           bool equation) {
  // Y standardization
  auto [y, ym, ysd] = standardize(data_y);

  // X standardization & polynomization
  Matrix x = xt::eval(generate_polynomial(data_x, p_degree));

  Matrix b;
  if (equation) {
    // calculate parameters witn normal equation
    auto xt = xt::transpose(x);
    b = xt::linalg::dot(
        xt::linalg::dot(xt::linalg::inv(xt::linalg::dot(xt, x)), xt), y);
    auto cost = (xt::sum(xt::pow(y - xt::linalg::dot(x, b), 2.f)) /
                 static_cast<DType>(x.shape()[0]))[0];
    std::cout << "calculated cost : " << cost << std::endl;
  } else {
    // learn parameters with Gradient Descent
    b = bgd(x, y, 15);
  }

  // create model
  auto model = [b, ym, ysd, p_degree](const auto& data_x) {
    auto x = xt::eval(generate_polynomial(data_x, p_degree));
    Matrix yhat = xt::linalg::dot(x, b);
    // restore scaling for predicted line values

    yhat = (yhat * ysd) + ym;
    return yhat;
  };
  return model;
}

int main() {
  // Download the data
  const std::string data_path{"web_traffic.tsv"};
  if (!fs::exists(data_path)) {
    const std::string data_url{
        R"(https://raw.githubusercontent.com/luispedro/BuildingMachineLearningSystemsWithPython/master/ch01/data/web_traffic.tsv)"};
    if (!utils::DownloadFile(data_url, data_path)) {
      std::cerr << "Unable to download the file " << data_url << std::endl;
      return 1;
    }
  }

  // Read the data
  io::CSVReader<2, io::trim_chars<' '>, io::no_quote_escape<'\t'>> data_tsv(
      data_path);

  std::vector<DType> raw_data_x;
  std::vector<DType> raw_data_y;

  bool done = false;
  do {
    try {
      DType x = 0, y = 0;
      done = !data_tsv.read_row(x, y);
      if (!done) {
        raw_data_x.push_back(x);
        raw_data_y.push_back(y);
      }
    } catch (const io::error::no_digit& err) {
      // ignore bad formated samples
      std::cout << err.what() << std::endl;
    }
  } while (!done);

  // shuffle data
  size_t seed = 3465467546;
  std::shuffle(raw_data_x.begin(), raw_data_x.end(),
               std::default_random_engine(seed));
  std::shuffle(raw_data_y.begin(), raw_data_y.end(),
               std::default_random_engine(seed));

  // map data to the tensor
  size_t rows = raw_data_x.size();
  auto shape_x = std::vector<size_t>{rows};
  const auto data_x = xt::adapt(raw_data_x, shape_x);
  std::cout << "X shape " << data_x.shape() << std::endl;

  auto shape_y = std::vector<size_t>{rows};
  const auto data_y = xt::adapt(raw_data_y, shape_y);
  std::cout << "Y shape " << data_y.shape() << std::endl;

  // generate new data
  auto minmax = xt::eval(xt::minmax(data_x));
  const Matrix new_x = xt::linspace<DType>(minmax[0][0], minmax[0][1], 2000);

  // poly line
  auto poly_model_eq = make_regression_model(data_x, data_y, 64, true);
  Matrix poly_line_values_eq = poly_model_eq(new_x);

  // poly line
  auto poly_model = make_regression_model(data_x, data_y, 10, false);
  Matrix poly_line_values = poly_model(new_x);

  // straight line
  auto line_model = make_regression_model(data_x, data_y, 2, false);
  Matrix line_values = line_model(new_x);

  // create adaptors with STL like interfaces
  auto x_coord = xt::view(new_x, xt::all());
  auto line = xt::view(line_values, xt::all());
  auto polyline = xt::view(poly_line_values, xt::all());
  auto polyline_eq = xt::view(poly_line_values_eq, xt::all());

  // plot the data we read and approximate
  plotcpp::Plot plt(true);
  plt.SetTerminal("qt");
  plt.SetTitle("Web traffic over the last month");
  plt.SetXLabel("Time");
  plt.SetYLabel("Hits/hour");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto time_range = minmax[0][1] - minmax[0][0];
  auto tic_size = 7 * 24;
  auto time_tics = time_range / tic_size;
  plt.SetXRange(-tic_size / 2, minmax[0][1] + tic_size / 2);

  plotcpp::Plot::Tics xtics;
  for (size_t t = 0; t < time_tics; ++t) {
    xtics.push_back({"week " + std::to_string(t), t * tic_size});
  }
  plt.SetXTics(xtics);

  plt.Draw2D(plotcpp::Points(data_x.begin(), data_x.end(), data_y.begin(),
                             "data", "lc rgb 'black' pt 1"),
             plotcpp::Lines(x_coord.begin(), x_coord.end(), line.begin(),
                            "line approx", "lc rgb 'red' lw 2"),
             plotcpp::Lines(x_coord.begin(), x_coord.end(), polyline.begin(),
                            "poly line approx d = 10", "lc rgb 'cyan' lw 2"),
             plotcpp::Lines(x_coord.begin(), x_coord.end(), polyline_eq.begin(),
                            "poly line approx d = 64", "lc rgb 'green' lw 2"));
  plt.Flush();

  return 0;
}
