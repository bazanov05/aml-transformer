#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::vector and std::string to Python lists and strings
#include "aml_tokenizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(aml_tokenizer, m) {
    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<int>(), py::arg("num_merges"))
        .def("train", &Tokenizer::train, py::arg("text"))
        .def("encode", &Tokenizer::encode, py::arg("text"))
        .def("decode", &Tokenizer::decode, py::arg("ids"))
        .def("save", &Tokenizer::save, py::arg("filepath"))
        .def("load", &Tokenizer::load, py::arg("filepath"))
        .def("get_vocab_size", &Tokenizer::get_vocab_size);
}