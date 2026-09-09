// telemetry.cpp
#include <nlohmann/json.hpp>
#include <fitsio.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include "baldr.h"   // Contains rtc_config and bdr_telem definition
#include <mutex>
#include <filesystem> //C++ 17

#include <functional>
#include <cstdio>


// long unsigned int telemetry_cnt=0;
// // Add local (to telemetry) variables here

// // Initialise variables assocated with the RTC.
// void initialise_telemetry(){
//     // Add initialisation code here
// }

using json = nlohmann::json;


extern std::mutex telemetry_mutex;


// A helper function to convert telemetry data into JSON.
// For example, here we assume each telemetry field is a ring buffer of Eigen::VectorXd,
// and for the timestamp we have a ring buffer of doubles.


//forward declaration
// adding in to allow generic field saves in probe method (which is different to dumping/saving telemetry buffer )
// Extractor signature: given a field name, fill M with (rows = steps, cols = field_len).
using FieldExtractor = std::function<bool(std::string_view /*field*/,
                                          Eigen::MatrixXd& /*M*/,
                                          std::string& /*why_not*/)>;

bool write_fields_to_fits(const std::string& path,
                          const std::vector<std::string>& fields,
                          FieldExtractor extract,
                          long nsteps,
                          const std::vector<std::pair<std::string, std::string>>& header_strs,
                          const std::vector<std::pair<std::string, long>>&    header_longs,
                          const std::vector<std::pair<std::string, double>>&  header_doubles);

/// we define redefine this late so should delete after testing
// using FieldExtractor = std::function<bool(std::string_view, Eigen::MatrixXd&, std::string&)>;

// bool write_fields_to_fits(const std::string& path,
//                           const std::vector<std::string>& fields,
//                           FieldExtractor extract,
//                           long nsteps);

static bool telem_extract_matrix(const bdr_telem& telem,
                                 std::string_view field,
                                 Eigen::MatrixXd& M,
                                 std::string& why)
{
    const std::size_t N = telem.timestamp.size(); // rows = number of samples

    auto fill_scalar_double = [&](const boost::circular_buffer<double>& buf) -> bool {
        if (buf.size() != N) { why = "inconsistent scalar<double> buffer size"; return false; }
        M.resize(static_cast<Eigen::Index>(N), 1);
        for (std::size_t i = 0; i < N; ++i) M(static_cast<Eigen::Index>(i), 0) = buf[i];
        return true;
    };
    auto fill_scalar_int = [&](const boost::circular_buffer<int>& buf) -> bool {
        if (buf.size() != N) { why = "inconsistent scalar<int> buffer size"; return false; }
        M.resize(static_cast<Eigen::Index>(N), 1);
        for (std::size_t i = 0; i < N; ++i) M(static_cast<Eigen::Index>(i), 0) = static_cast<double>(buf[i]);
        return true;
    };
    auto fill_vector = [&](const boost::circular_buffer<Eigen::VectorXd>& buf) -> bool {
        if (buf.size() != N) { why = "inconsistent vector buffer size"; return false; }
        if (N == 0) { M.resize(0, 0); return true; }
        const Eigen::Index L = static_cast<Eigen::Index>(buf[0].size());
        M.resize(static_cast<Eigen::Index>(N), L);
        for (std::size_t i = 0; i < N; ++i) {
            if (buf[i].size() != L) { why = "ragged vector lengths"; return false; }
            M.row(static_cast<Eigen::Index>(i)) = buf[i].transpose(); // row = step
        }
        return true;
    };

    if      (field == "timestamp")     return fill_scalar_double(telem.timestamp);
    else if (field == "LO_servo_mode") return fill_scalar_int(telem.LO_servo_mode);
    else if (field == "HO_servo_mode") return fill_scalar_int(telem.HO_servo_mode);
    else if (field == "img")           return fill_vector(telem.img);
    else if (field == "img_dm")        return fill_vector(telem.img_dm);
    else if (field == "signal")        return fill_vector(telem.signal);
    else if (field == "e_LO")          return fill_vector(telem.e_LO);
    else if (field == "u_LO")          return fill_vector(telem.u_LO);
    else if (field == "e_HO")          return fill_vector(telem.e_HO);
    else if (field == "u_HO")          return fill_vector(telem.u_HO);
    else if (field == "c_LO")          return fill_vector(telem.c_LO);
    else if (field == "c_HO")          return fill_vector(telem.c_HO);
    else if (field == "c_inj")         return fill_vector(telem.c_inj);
    else if (field == "rmse_est")      return fill_scalar_double(telem.rmse_est);
    else if (field == "snr")           return fill_scalar_double(telem.snr);

    why = "unknown field";
    return false;
}


// //refactor that i wont do for now 
// bool write_telemetry_to_fits(const std::string& path,
//                              const bdr_telem& telem,
//                              const std::vector<std::string>& fields)
// {
//     auto extractor = [&](std::string_view name, Eigen::MatrixXd& M, std::string& why) {
//         return telem_extract_matrix(telem, name, M, why);
//     };
//     const long nsteps = static_cast<long>(telem.timestamp.size());
//     return write_fields_to_fits(path, fields, extractor, nsteps);
// }

// // Convenience overload: write ALL numeric telemetry fields
// bool write_telemetry_to_fits(const std::string& path, const bdr_telem& telem)
// {
//     std::vector<std::string> fields;
//     fields.reserve(bdr_telem::kNumericSavableFields.size());
//     for (auto sv : bdr_telem::kNumericSavableFields) fields.emplace_back(sv);
//     return write_telemetry_to_fits(path, telem, fields);
// }

// Extract the last N samples of a telemetry field into a matrix M (rows=N, cols=field_len).
// Thread-safe: grabs telemetry_mutex while reading.
bool telem_extract_matrix_lastN(const bdr_telem& telem,
                                std::mutex& telemetry_mutex,
                                std::string_view field,
                                std::size_t N,
                                Eigen::MatrixXd& M,
                                std::string& why)
{
    std::lock_guard<std::mutex> lk(telemetry_mutex);

    auto fill_scalar_double_lastN = [&](const boost::circular_buffer<double>& buf)->bool{
        const std::size_t have = buf.size();
        const std::size_t take = std::min(N, have);
        const std::size_t start = have - take;
        M.resize(static_cast<Eigen::Index>(take), 1);
        for (std::size_t i = 0; i < take; ++i) M(static_cast<Eigen::Index>(i), 0) = buf[start + i];
        return true;
    };
    auto fill_scalar_int_lastN = [&](const boost::circular_buffer<int>& buf)->bool{
        const std::size_t have = buf.size();
        const std::size_t take = std::min(N, have);
        const std::size_t start = have - take;
        M.resize(static_cast<Eigen::Index>(take), 1);
        for (std::size_t i = 0; i < take; ++i) M(static_cast<Eigen::Index>(i), 0) = static_cast<double>(buf[start + i]);
        return true;
    };
    auto fill_vector_lastN = [&](const boost::circular_buffer<Eigen::VectorXd>& buf)->bool{
        const std::size_t have = buf.size();
        const std::size_t take = std::min(N, have);
        const std::size_t start = have - take;
        if (take == 0) { M.resize(0,0); return true; }
        const Eigen::Index L = static_cast<Eigen::Index>(buf[start].size());
        M.resize(static_cast<Eigen::Index>(take), L);
        for (std::size_t i = 0; i < take; ++i) {
            const auto& v = buf[start + i];
            if (v.size() != L) { why = "ragged vector lengths"; return false; }
            M.row(static_cast<Eigen::Index>(i)) = v.transpose();
        }
        return true;
    };

    if      (field == "timestamp")     return fill_scalar_double_lastN(telem.timestamp);
    else if (field == "LO_servo_mode") return fill_scalar_int_lastN(telem.LO_servo_mode);
    else if (field == "HO_servo_mode") return fill_scalar_int_lastN(telem.HO_servo_mode);
    else if (field == "img")           return fill_vector_lastN(telem.img);
    else if (field == "img_dm")        return fill_vector_lastN(telem.img_dm);
    else if (field == "signal")        return fill_vector_lastN(telem.signal);
    else if (field == "e_LO")          return fill_vector_lastN(telem.e_LO);
    else if (field == "u_LO")          return fill_vector_lastN(telem.u_LO);
    else if (field == "e_HO")          return fill_vector_lastN(telem.e_HO);
    else if (field == "u_HO")          return fill_vector_lastN(telem.u_HO);
    else if (field == "c_LO")          return fill_vector_lastN(telem.c_LO);
    else if (field == "c_HO")          return fill_vector_lastN(telem.c_HO);
    else if (field == "c_inj")         return fill_vector_lastN(telem.c_inj);
    else if (field == "rmse_est")      return fill_scalar_double_lastN(telem.rmse_est);
    else if (field == "snr")           return fill_scalar_double_lastN(telem.snr);

    why = "unknown field";
    return false;
}
////


json telemetry_to_json(const bdr_telem &telemetry) {
    json j;
    j["counter"] = telemetry.counter;

    // Convert timestamps:
    json ts = json::array();
    for (const double t : telemetry.timestamp) {
        ts.push_back(t);
    }
    j["timestamps"] = ts;

    j["LO_servo_mode"] = telemetry.LO_servo_mode;
    j["HO_servo_mode"] = telemetry.HO_servo_mode;

    // For each telemetry field that is a ring buffer of Eigen::VectorXd, convert it.
    auto eigen_vec_to_json = [](const boost::circular_buffer<Eigen::VectorXd>& buff) -> json {
        json arr = json::array();
        for (const auto &v : buff) {
            // Convert Eigen::VectorXd to std::vector<double>
            std::vector<double> vec(v.data(), v.data() + v.size());
            arr.push_back(vec);
        }
        return arr;
    };

    j["img"] = eigen_vec_to_json(telemetry.img);
    j["img_dm"] = eigen_vec_to_json(telemetry.img_dm);
    j["signal"] = eigen_vec_to_json(telemetry.signal);
    j["e_LO"] = eigen_vec_to_json(telemetry.e_LO);
    j["u_LO"] = eigen_vec_to_json(telemetry.u_LO);
    j["e_HO"] = eigen_vec_to_json(telemetry.e_HO);
    j["u_HO"] = eigen_vec_to_json(telemetry.u_HO);
    j["c_LO"] = eigen_vec_to_json(telemetry.c_LO);
    j["c_HO"] = eigen_vec_to_json(telemetry.c_HO);
    j["c_inj"] = eigen_vec_to_json(telemetry.c_inj);     // NEW:
    j["rmse_est"] = telemetry.rmse_est;
    j["snr"] = telemetry.snr;


    return j;
}


// Function that writes the telemetry (bdr_telem) to a FITS file.
// It writes a binary table with the following columns:
//   1. COUNTER (integer)
//   2. TIMESTAMPS (double)
//   3. LO_SERVO_MODE (integer)
//   4. HO_SERVO_MODE (integer)
//   5. IMG       (Eigen::VectorXd flattened)
//   6. IMG_DM    (Eigen::VectorXd flattened)
//   7. SIGNAL    (Eigen::VectorXd flattened)
//   8. E_LO      (Eigen::VectorXd flattened)
//   9. U_LO      (Eigen::VectorXd flattened)
//  10. E_HO      (Eigen::VectorXd flattened)
//  11. U_HO      (Eigen::VectorXd flattened)
//  12. C_LO      (Eigen::VectorXd flattened)
//  13. C_HO      (Eigen::VectorXd flattened)
//  14. C_INJ     (Eigen::VectorXd flattened)
//  15. rmse_est   <double>
//. 16. snr        <double>

// It creates a binary table with 16 columns: COUNTER, TIMESTAMPS,
// LO_SERVO_MODE, HO_SERVO_MODE, IMG, IMG_DM, SIGNAL, E_LO, U_LO, E_HO, U_HO, C_LO, C_HO, C_INJ, 
int write_telemetry_to_fits(const bdr_telem &telemetry, const std::string &filename) {
    fitsfile *fptr = nullptr;
    int status = 0;
    
    long nrows = telemetry.timestamp.size();
    if (nrows == 0) {
        std::cerr << "No telemetry data available to write." << std::endl;
        return 0;
    }
    
    auto getVecLength = [](const boost::circular_buffer<Eigen::VectorXd> &buff) -> long {
        return (buff.empty() ? 0 : buff.front().size());
    };
    
    long len_img    = getVecLength(telemetry.img);
    long len_img_dm = getVecLength(telemetry.img_dm);
    long len_signal = getVecLength(telemetry.signal);
    long len_e_LO   = getVecLength(telemetry.e_LO);
    long len_u_LO   = getVecLength(telemetry.u_LO);
    long len_e_HO   = getVecLength(telemetry.e_HO);
    long len_u_HO   = getVecLength(telemetry.u_HO);
    long len_c_LO   = getVecLength(telemetry.c_LO);
    long len_c_HO   = getVecLength(telemetry.c_HO);
    long len_c_inj  = getVecLength(telemetry.c_inj); // NEW

    auto makeFormat = [](long len) -> std::string {
        return std::to_string(len) + "D";
    };
    
    //const int ncols = 13;
    
    const int ncols = 16; // +1 for C_INJ //const int ncols = 15;  
    char *ttype[ncols] = {
        const_cast<char*>("COUNTER"),
        const_cast<char*>("TIMESTAMPS"),
        const_cast<char*>("LO_SERVO_MODE"),
        const_cast<char*>("HO_SERVO_MODE"),
        const_cast<char*>("IMG"),
        const_cast<char*>("IMG_DM"),
        const_cast<char*>("SIGNAL"),
        const_cast<char*>("E_LO"),
        const_cast<char*>("U_LO"),
        const_cast<char*>("E_HO"),
        const_cast<char*>("U_HO"),
        const_cast<char*>("C_LO"),
        const_cast<char*>("C_HO"),
        const_cast<char*>("C_INJ"),     // NEW (col 14)
        const_cast<char*>("RMSE_EST"), //<--- New
        const_cast<char*>("SNR")     
    };
    
    char tform[ncols][20];
    snprintf(tform[0], sizeof(tform[0]), "1J");
    snprintf(tform[1], sizeof(tform[1]), "1D");
    snprintf(tform[2], sizeof(tform[2]), "1J");
    snprintf(tform[3], sizeof(tform[3]), "1J");
    snprintf(tform[4], sizeof(tform[4]), "%s", makeFormat(len_img).c_str());
    snprintf(tform[5], sizeof(tform[5]), "%s", makeFormat(len_img_dm).c_str());
    snprintf(tform[6], sizeof(tform[6]), "%s", makeFormat(len_signal).c_str());
    snprintf(tform[7], sizeof(tform[7]), "%s", makeFormat(len_e_LO).c_str());
    snprintf(tform[8], sizeof(tform[8]), "%s", makeFormat(len_u_LO).c_str());
    snprintf(tform[9], sizeof(tform[9]), "%s", makeFormat(len_e_HO).c_str());
    snprintf(tform[10], sizeof(tform[10]), "%s", makeFormat(len_u_HO).c_str());
    snprintf(tform[11], sizeof(tform[11]), "%s", makeFormat(len_c_LO).c_str());
    snprintf(tform[12], sizeof(tform[12]), "%s", makeFormat(len_c_HO).c_str());
    snprintf(tform[13], sizeof(tform[13]), "%s", makeFormat(len_c_inj).c_str()); // NEW (C_INJ)
    snprintf(tform[14], sizeof(tform[14]), "1D");  // RMSE_EST
    snprintf(tform[15], sizeof(tform[15]), "1D");  // SNR

    char* tform_ptr[ncols];
    for (int i = 0; i < ncols; ++i) {
        tform_ptr[i] = tform[i];
    }
    
    char *tunit[ncols] = {
        const_cast<char*>(""),
        const_cast<char*>("microsec"),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>(""),
        const_cast<char*>(""), 
        const_cast<char*>(""),
        const_cast<char*>(""),         /// C_INJ
        const_cast<char*>(""),        // 15 RMSE_EST
        const_cast<char*>("")         // 16 SNR
    };
    
    if (fits_create_file(&fptr, ("!" + filename).c_str(), &status))
        throw std::runtime_error("Error creating FITS file");

    if (fits_create_tbl(fptr, BINARY_TBL, nrows, ncols, ttype, tform_ptr, tunit, "TELEMETRY", &status))
        throw std::runtime_error("Error creating telemetry table");

    // Prepare scalar columns
    std::vector<int> counterCol(nrows);
    for (long i = 0; i < nrows; ++i)
        counterCol[i] = i;

    std::vector<double> timestamps(telemetry.timestamp.begin(), telemetry.timestamp.end());
    std::vector<int> loServo(telemetry.LO_servo_mode.begin(), telemetry.LO_servo_mode.end());
    std::vector<int> hoServo(telemetry.HO_servo_mode.begin(), telemetry.HO_servo_mode.end());


    // Write scalar columns
    if (fits_write_col(fptr, TINT, 1, 1, 1, nrows, counterCol.data(), &status))
        throw std::runtime_error("Error writing COUNTER column");
    if (fits_write_col(fptr, TDOUBLE, 2, 1, 1, nrows, timestamps.data(), &status))
        throw std::runtime_error("Error writing TIMESTAMPS column");
    if (fits_write_col(fptr, TINT, 3, 1, 1, nrows, loServo.data(), &status))
        throw std::runtime_error("Error writing LO_SERVO_MODE column");
    if (fits_write_col(fptr, TINT, 4, 1, 1, nrows, hoServo.data(), &status))
        throw std::runtime_error("Error writing HO_SERVO_MODE column");

    
    // Helper: convert circular buffer to vector<vector<double>>
    auto flattenBuffer = [](const boost::circular_buffer<Eigen::VectorXd>& buff) -> std::vector<std::vector<double>> {
        std::vector<std::vector<double>> flat;
        flat.reserve(buff.size());
        for (const auto& v : buff) {
            flat.emplace_back(v.data(), v.data() + v.size());
        }
        return flat;
    };

    auto imgCol    = flattenBuffer(telemetry.img);
    auto img_dmCol = flattenBuffer(telemetry.img_dm);
    auto signalCol = flattenBuffer(telemetry.signal);
    auto e_LOCol   = flattenBuffer(telemetry.e_LO);
    auto u_LOCol   = flattenBuffer(telemetry.u_LO);
    auto e_HOCol   = flattenBuffer(telemetry.e_HO);
    auto u_HOCol   = flattenBuffer(telemetry.u_HO);
    auto c_LOCol   = flattenBuffer(telemetry.c_LO);
    auto c_HOCol   = flattenBuffer(telemetry.c_HO);
    auto c_injCol  = flattenBuffer(telemetry.c_inj); // NEW

    // Helper: write per-row vector
    auto writeVectorColumn = [&](int colnum, const std::vector<std::vector<double>>& buffer) {
        for (long i = 0; i < nrows; ++i) {
            if (fits_write_col(fptr, TDOUBLE, colnum, i + 1, 1, buffer[i].size(), (void*) buffer[i].data(), &status))
                throw std::runtime_error("Error writing vector column");
        }
    };

    // Write vector columns correctly!
    writeVectorColumn(5, imgCol);
    writeVectorColumn(6, img_dmCol);
    writeVectorColumn(7, signalCol);
    writeVectorColumn(8, e_LOCol);
    writeVectorColumn(9, u_LOCol);
    writeVectorColumn(10, e_HOCol);
    writeVectorColumn(11, u_HOCol);
    writeVectorColumn(12, c_LOCol);
    writeVectorColumn(13, c_HOCol);
    writeVectorColumn(14, c_injCol); // NEW

    // RMSE_EST and SNR shift by +1
    std::vector<double> rmse_est_vec(telemetry.rmse_est.begin(), telemetry.rmse_est.end());
    if (fits_write_col(fptr, TDOUBLE, 15, 1, 1, nrows, rmse_est_vec.data(), &status))
        throw std::runtime_error("Error writing RMSE_EST column");

    std::vector<double> snr_vec(telemetry.snr.begin(), telemetry.snr.end());
    if (fits_write_col(fptr, TDOUBLE, 16, 1, 1, nrows, snr_vec.data(), &status))
        throw std::runtime_error("Error writing SNR column");

    
    // ================== Append reference snapshots as image HDUs ==================
    {
        // Snapshot the current references under the telemetry/config mutex
        Eigen::VectorXd I0, N0, DARK;
        {
            std::lock_guard<std::mutex> lk(telemetry_mutex);
            I0   = rtc_config.I0_dm_runtime;          // DM-space reference (double)
            N0   = rtc_config.N0_dm_runtime;          // DM-space normalization ref
            DARK = rtc_config.reduction.dark;         // per-pixel dark (raw ADU)
        }

        auto write_vector_image_ext = [&](const char* extname,
                                        const Eigen::VectorXd& v) -> void {
            if (v.size() <= 0) return;

            // Create a 1D DOUBLE image [len]
            int status_local = 0;
            long naxes[1]    = { static_cast<long>(v.size()) };

            if (fits_create_img(fptr, DOUBLE_IMG, 1, naxes, &status_local) || status_local) {
                fits_report_error(stderr, status_local);
                throw std::runtime_error(std::string("Error creating image HDU for ") + extname);
            }

            // Label the extension
            if (fits_update_key(fptr, TSTRING, const_cast<char*>("EXTNAME"),
                                (void*)extname, nullptr, &status_local) || status_local) {
                fits_report_error(stderr, status_local);
                throw std::runtime_error(std::string("Error setting EXTNAME for ") + extname);
            }

            // Optional: annotate length
            long vlen = static_cast<long>(v.size());
            if (fits_update_key(fptr, TLONG, const_cast<char*>("VLEN"),
                                &vlen, const_cast<char*>("vector length"), &status_local) || status_local) {
                fits_report_error(stderr, status_local);
                throw std::runtime_error(std::string("Error writing VLEN for ") + extname);
            }

            // Write data
            std::vector<double> buf(v.data(), v.data() + v.size());
            long firstelem = 1;
            long nelem     = vlen;
            if (fits_write_img(fptr, TDOUBLE, firstelem, nelem, buf.data(), &status_local) || status_local) {
                fits_report_error(stderr, status_local);
                throw std::runtime_error(std::string("Error writing data for ") + extname);
            }
        };

        // I0 and N0 are DM-space (length ~140 or 144). DARK is raw-pixel (len_img).
        write_vector_image_ext("I0_DM_REF",   I0);
        write_vector_image_ext("N0_DM_REF",   N0);
        write_vector_image_ext("DARK_REF",    DARK);
    }
    // ============================================================================


    
    // Close the file
    if (fits_close_file(fptr, &status)) {
        fits_report_error(stderr, status);
        throw std::runtime_error("Error closing FITS file");
    }

    std::cout << "Telemetry successfully written to " << filename << std::endl;
    return 0;
}




/**
 * Write a single FITS file containing one IMAGE extension (HDU) per field.
 * - Primary HDU: empty (0-D) image; carries global metadata (e.g., NSTEPS).
 * - For each field:
 *     EXTNAME=<field>, NSTEP=<rows>, NCOLS=<cols>
 *     Data written as DOUBLE_IMG with CFITSIO dims [cols, rows].
 *
 * @param path   Output .fits path. Existing file is clobbered.
 * @param fields Ordered list of field names to write (each becomes one HDU).
 * @param extract Callback that materializes (rows x cols) matrix for a field.
 * @param nsteps  For global metadata only (rows sanity).
 *
 * @return true on success; false on any CFITSIO or extractor error.
 */

bool write_fields_to_fits(const std::string& path,
                          const std::vector<std::string>& fields,
                          FieldExtractor extract,
                          long nsteps,
                          const std::vector<std::pair<std::string, std::string>>& header_strs,
                          const std::vector<std::pair<std::string, long>>&    header_longs,
                          const std::vector<std::pair<std::string, double>>&  header_doubles)
{
    int status = 0;
    fitsfile* f = nullptr;

    std::string fname = "!" + path;
    fits_create_file(&f, fname.c_str(), &status);
    if (status) { fits_report_error(stderr, status); return false; }

    // Primary HDU (no data)
    fits_create_img(f, DOUBLE_IMG, 0, nullptr, &status);
    if (status) { fits_report_error(stderr, status); fits_close_file(f, &status); return false; }

    // Global keys
    fits_write_key(f, TLONG, "NSTEPS", &nsteps, (char*)"rows = steps for matrices", &status);

    // NEW: write extra primary-HDU headers
    for (const auto& kv : header_strs) {
        const char* key = kv.first.c_str();
        const char* val = kv.second.c_str();
        fits_write_key(f, TSTRING, const_cast<char*>(key), (void*)val, nullptr, &status);
        if (status) { fits_report_error(stderr, status); fits_close_file(f, &status); return false; }
    }
    for (const auto& kv : header_longs) {
        const char* key = kv.first.c_str();
        long        v   = kv.second;
        fits_write_key(f, TLONG, const_cast<char*>(key), (void*)&v, nullptr, &status);
        if (status) { fits_report_error(stderr, status); fits_close_file(f, &status); return false; }
    }
    for (const auto& kv : header_doubles) {
        const char* key = kv.first.c_str();
        double      v   = kv.second;
        fits_write_key(f, TDOUBLE, const_cast<char*>(key), (void*)&v, nullptr, &status);
        if (status) { fits_report_error(stderr, status); fits_close_file(f, &status); return false; }
    }

    // ... keep your loop that creates one IMAGE HDU per field (EXTNAME/NSTEP/NCOLS + data) ...
    for (const auto& name : fields) {
        Eigen::MatrixXd M;
        std::string why;
        if (!extract(name, M, why)) {
            std::fprintf(stderr, "write_fields_to_fits: cannot extract '%s': %s\n",
                         name.c_str(), why.c_str());
            fits_close_file(f, &status);
            return false;
        }
        const long rows = static_cast<long>(M.rows());
        const long cols = static_cast<long>(M.cols());
        long naxes[2] = { cols, rows };
        fits_create_img(f, DOUBLE_IMG, 2, naxes, &status);
        if (status) { fits_report_error(stderr, status); fits_close_file(f, &status); return false; }

        char extname[72]; std::snprintf(extname, sizeof(extname), "%s", name.c_str());
        fits_write_key(f, TSTRING, "EXTNAME", extname, (char*)"telemetry field", &status);
        fits_write_key(f, TLONG,   "NSTEP",   (void*)&rows, (char*)"rows = steps", &status);
        fits_write_key(f, TLONG,   "NCOLS",   (void*)&cols, (char*)"cols = field length", &status);

        const long nelem = static_cast<long>(M.size());
        fits_write_img(f, TDOUBLE, 1, nelem, const_cast<double*>(M.data()), &status);
        if (status) { fits_report_error(stderr, status); fits_close_file(f, &status); return false; }
    }

    fits_close_file(f, &status);
    if (status) { fits_report_error(stderr, status); return false; }
    return true;
}


// bool write_fields_to_fits(const std::string& path,
//                           const std::vector<std::string>& fields,
//                           FieldExtractor extract,
//                           long nsteps)
// {
//     int status = 0;
//     fitsfile* f = nullptr;

//     // "!" prefix: overwrite existing file
//     std::string fname = "!" + path;
//     fits_create_file(&f, fname.c_str(), &status);
//     if (status) { fits_report_error(stderr, status); return false; }

//     // Primary HDU (no data). Add global metadata here.
//     fits_create_img(f, DOUBLE_IMG, 0, nullptr, &status);
//     if (status) { fits_report_error(stderr, status); fits_close_file(f, &status); return false; }

//     // Global keys (optional: add CREATOR, DATE, etc.)
//     fits_write_key(f, TLONG, "NSTEPS", &nsteps, (char*)"rows = steps for matrices", &status);

//     for (const auto& name : fields) {
//         Eigen::MatrixXd M;
//         std::string why;
//         if (!extract(name, M, why)) {
//             std::fprintf(stderr, "write_fields_to_fits: cannot extract '%s': %s\n",
//                          name.c_str(), why.c_str());
//             fits_close_file(f, &status);
//             return false;
//         }

//         // CFITSIO expects dims as [NAXIS1, NAXIS2] = [cols, rows]
//         const long rows = static_cast<long>(M.rows());
//         const long cols = static_cast<long>(M.cols());

//         long naxes[2] = { cols, rows };
//         fits_create_img(f, DOUBLE_IMG, 2, naxes, &status);
//         if (status) { fits_report_error(stderr, status); fits_close_file(f, &status); return false; }

//         // Label the extension and store simple per-field metadata
//         char extname[72]; std::snprintf(extname, sizeof(extname), "%s", name.c_str());
//         fits_write_key(f, TSTRING, "EXTNAME", extname, (char*)"telemetry field", &status);
//         fits_write_key(f, TLONG,   "NSTEP",   (void*)&rows, (char*)"rows = steps", &status);
//         fits_write_key(f, TLONG,   "NCOLS",   (void*)&cols, (char*)"cols = field length", &status);

//         // Eigen defaults to column-major, contiguous: safe to write directly
//         const long nelem = static_cast<long>(M.size());
//         fits_write_img(f, TDOUBLE, 1, nelem, const_cast<double*>(M.data()), &status);
//         if (status) { fits_report_error(stderr, status); fits_close_file(f, &status); return false; }
//     }

//     fits_close_file(f, &status);
//     if (status) { fits_report_error(stderr, status); return false; }
//     return true;
// }


void telemetry(){
    // Telemetry thread loop.
    while (servo_mode != SERVO_STOP) {
        // Sleep for a fixed interval (e.g., 1 second) between telemetry writes.
        std::this_thread::sleep_for(std::chrono::seconds(1));

        if (rtc_config.state.take_telemetry==1) {
            // Get current system time for file naming.
            auto now = std::chrono::system_clock::now();                
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            std::tm tm = *std::localtime(&t);

            std::filesystem::create_directories(telem_save_path);

            std::ostringstream oss;
            oss << telem_save_path << "telemetry_" << std::put_time(&tm, "%Y%m%d_%H%M%S");

            if (telemFormat == "fits") {
                oss << ".fits";
            } else {
                oss << ".json";
            }
            std::string filename = oss.str();

            // if (telemFormat == "fits") {
            //     oss << "/home/asg/Music/telemetry_" 
            //         << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".fits";
            // } else { // default to "json"
            //     oss << "/home/asg/Music/telemetry_" 
            //         << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".json";
            // }
            // std::string filename = oss.str();

            bdr_telem currentTelem;
            {   // Lock telemetry mutex and copy current telemetry.
                std::lock_guard<std::mutex> lock(telemetry_mutex);
                currentTelem = rtc_config.telem;
            }
            
            if (telemFormat == "fits") {
                // Call your FITS writer.
                try {
                    int status = write_telemetry_to_fits(currentTelem, filename);
                    std::cout << "Telemetry written to FITS file: " << filename << "with output" << status << std::endl;
                } catch (const std::exception &ex) {
                    std::cerr << "Error writing telemetry FITS file: " << ex.what() << std::endl;
                }
            } else {
                // Convert telemetry to JSON and write it.
                json j = telemetry_to_json(currentTelem);
                std::ofstream ofs(filename);
                if (ofs.is_open()) {
                    ofs << j.dump(4);
                    ofs.close();
                    std::cout << "Telemetry written to " << filename << std::endl;
                } else {
                    std::cerr << "Error opening file " << filename << " for writing." << std::endl;
                }
            }
            // Reset the take_telemetry flag.
            rtc_config.state.take_telemetry = 0;
        }
    }
}

// reading the vector in bdr_telem that lists the numeric telemetry fields
std::vector<std::string> list_all_numeric_telem_fields() {
    std::vector<std::string> out;
    out.reserve(bdr_telem::kNumericSavableFields.size());
    for (std::string_view sv : bdr_telem::kNumericSavableFields) {
        out.emplace_back(sv);
    }
    return out;
}


/// previous to for_ben


// // telemetry.cpp
// #include <nlohmann/json.hpp>
// #include <fitsio.h>
// #include <fstream>
// #include <sstream>
// #include <chrono>
// #include <vector>
// #include <string>
// #include <iostream>
// #include <stdexcept>
// #include <iomanip>
// #include "baldr.h"   // Contains rtc_config and bdr_telem definition
// #include <mutex>


// // long unsigned int telemetry_cnt=0;
// // // Add local (to telemetry) variables here

// // // Initialise variables assocated with the RTC.
// // void initialise_telemetry(){
// //     // Add initialisation code here
// // }

// using json = nlohmann::json;


// extern std::mutex telemetry_mutex;


// // A helper function to convert telemetry data into JSON.
// // For example, here we assume each telemetry field is a ring buffer of Eigen::VectorXd,
// // and for the timestamp we have a ring buffer of doubles.
// json telemetry_to_json(const bdr_telem &telemetry) {
//     json j;
//     j["counter"] = telemetry.counter;

//     // Convert timestamps:
//     json ts = json::array();
//     for (const double t : telemetry.timestamp) {
//         ts.push_back(t);
//     }
//     j["timestamps"] = ts;

//     j["LO_servo_mode"] = telemetry.LO_servo_mode;
//     j["HO_servo_mode"] = telemetry.HO_servo_mode;

//     // For each telemetry field that is a ring buffer of Eigen::VectorXd, convert it.
//     auto eigen_vec_to_json = [](const boost::circular_buffer<Eigen::VectorXd>& buff) -> json {
//         json arr = json::array();
//         for (const auto &v : buff) {
//             // Convert Eigen::VectorXd to std::vector<double>
//             std::vector<double> vec(v.data(), v.data() + v.size());
//             arr.push_back(vec);
//         }
//         return arr;
//     };

//     j["img"] = eigen_vec_to_json(telemetry.img);
//     j["img_dm"] = eigen_vec_to_json(telemetry.img_dm);
//     j["signal"] = eigen_vec_to_json(telemetry.signal);
//     j["e_LO"] = eigen_vec_to_json(telemetry.e_LO);
//     j["u_LO"] = eigen_vec_to_json(telemetry.u_LO);
//     j["e_HO"] = eigen_vec_to_json(telemetry.e_HO);
//     j["u_HO"] = eigen_vec_to_json(telemetry.u_HO);
//     j["c_LO"] = eigen_vec_to_json(telemetry.c_LO);
//     j["c_HO"] = eigen_vec_to_json(telemetry.c_HO);

//     json rmse_arr = json::array();
//     for (const double val : telemetry.rmse_est) {
//         rmse_arr.push_back(val);
//     }
//     j["rmse_est"] = rmse_arr;

//     // Add SNR
//     json snr_arr = json::array();
//     for (const double val : telemetry.snr) {
//         snr_arr.push_back(val);
//     }
//     j["snr"] = snr_arr;

//     return j;
// }


// // Function that writes the telemetry (bdr_telem) to a FITS file.
// // It writes a binary table with the following columns:
// //   1. COUNTER (integer)
// //   2. TIMESTAMPS (double)
// //   3. LO_SERVO_MODE (integer)
// //   4. HO_SERVO_MODE (integer)
// //   5. IMG       (Eigen::VectorXd flattened)
// //   6. IMG_DM    (Eigen::VectorXd flattened)
// //   7. SIGNAL    (Eigen::VectorXd flattened)
// //   8. E_LO      (Eigen::VectorXd flattened)
// //   9. U_LO      (Eigen::VectorXd flattened)
// //  10. E_HO      (Eigen::VectorXd flattened)
// //  11. U_HO      (Eigen::VectorXd flattened)
// //  12. C_LO      (Eigen::VectorXd flattened)
// //  13. C_HO      (Eigen::VectorXd flattened)

// // It creates a binary table with 13 columns: COUNTER, TIMESTAMPS,
// // LO_SERVO_MODE, HO_SERVO_MODE, IMG, IMG_DM, SIGNAL, E_LO, U_LO, E_HO, U_HO, C_LO, and C_HO.
// int write_telemetry_to_fits(const bdr_telem &telemetry, const std::string &filename) {
//     fitsfile *fptr = nullptr;
//     int status = 0;
    
//     // Use the size of the timestamp ring buffer as the number of samples (rows).
//     long nrows = telemetry.timestamp.size();
//     if (nrows == 0) {
//         std::cerr << "No telemetry data available to write." << std::endl;
//         return 0;
//     }
    
//     // Helper lambda: get the length of an Eigen::VectorXd from a circular buffer.
//     auto getVecLength = [](const boost::circular_buffer<Eigen::VectorXd>& buff) -> long {
//         return (buff.empty() ? 0 : buff.front().size());
//     };
//     long len_img     = getVecLength(telemetry.img);
//     long len_img_dm  = getVecLength(telemetry.img_dm);
//     long len_signal  = getVecLength(telemetry.signal);
//     long len_e_LO    = getVecLength(telemetry.e_LO);
//     long len_u_LO    = getVecLength(telemetry.u_LO);
//     long len_e_HO    = getVecLength(telemetry.e_HO);
//     long len_u_HO    = getVecLength(telemetry.u_HO);
//     long len_c_LO    = getVecLength(telemetry.c_LO);
//     long len_c_HO    = getVecLength(telemetry.c_HO);
    
//     // Lambda to create a FITS format string for a fixed-length vector of doubles.
//     auto makeFormat = [=](long len) -> std::string {
//         return std::to_string(len) + "D"; // For example, "1024D"
//     };
//     std::string form_img     = makeFormat(len_img);
//     std::string form_img_dm  = makeFormat(len_img_dm);
//     std::string form_signal  = makeFormat(len_signal);
//     std::string form_e_LO    = makeFormat(len_e_LO);
//     std::string form_u_LO    = makeFormat(len_u_LO);
//     std::string form_e_HO    = makeFormat(len_e_HO);
//     std::string form_u_HO    = makeFormat(len_u_HO);
//     std::string form_c_LO    = makeFormat(len_c_LO);
//     std::string form_c_HO    = makeFormat(len_c_HO);
    
//     // Define the number of columns.
//     const int ncols = 15; //13;
    
//     // Define column names (TTYPE keywords).
//     char *ttype[ncols] = {
//         const_cast<char*>("COUNTER"),
//         const_cast<char*>("TIMESTAMPS"),
//         const_cast<char*>("LO_SERVO_MODE"),
//         const_cast<char*>("HO_SERVO_MODE"),
//         const_cast<char*>("IMG"),
//         const_cast<char*>("IMG_DM"),
//         const_cast<char*>("SIGNAL"),
//         const_cast<char*>("E_LO"),
//         const_cast<char*>("U_LO"),
//         const_cast<char*>("E_HO"),
//         const_cast<char*>("U_HO"),
//         const_cast<char*>("C_LO"),
//         const_cast<char*>("RMSE_EST"),
//         const_cast<char*>("SNR"),
//     };
    
//     // Create a 2D array for the TFORM keywords. Each row is a C string.
//     char tform[ncols][20];
//     snprintf(tform[0], sizeof(tform[0]), "1J");              // COUNTER: integer
//     snprintf(tform[1], sizeof(tform[1]), "1D");              // TIMESTAMPS: double
//     snprintf(tform[2], sizeof(tform[2]), "1J");              // LO_SERVO_MODE: integer
//     snprintf(tform[3], sizeof(tform[3]), "1J");              // HO_SERVO_MODE: integer
//     snprintf(tform[4], sizeof(tform[4]), "%s", form_img.c_str());
//     snprintf(tform[5], sizeof(tform[5]), "%s", form_img_dm.c_str());
//     snprintf(tform[6], sizeof(tform[6]), "%s", form_signal.c_str());
//     snprintf(tform[7], sizeof(tform[7]), "%s", form_e_LO.c_str());
//     snprintf(tform[8], sizeof(tform[8]), "%s", form_u_LO.c_str());
//     snprintf(tform[9], sizeof(tform[9]), "%s", form_e_HO.c_str());
//     snprintf(tform[10], sizeof(tform[10]), "%s", form_u_HO.c_str());
//     snprintf(tform[11], sizeof(tform[11]), "%s", form_c_LO.c_str());
//     snprintf(tform[12], sizeof(tform[12]), "%s", form_c_HO.c_str());
//     snprintf(tform[13], sizeof(tform[13]), "1D"); // RMSE_EST (one double per row)
//     snprintf(tform[14], sizeof(tform[14]), "1D"); // SNR (one double per row)
    
//     // Create an array of char* pointers, one for each column.
//     char* tform_ptr[ncols];
//     for (int i = 0; i < ncols; i++) {
//         tform_ptr[i] = tform[i];
//     }
    
//     // Define column units 
//     char *tunit[ncols] = {
//         const_cast<char*>(""),
//         const_cast<char*>("microsec"),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>(""),
//         const_cast<char*>("")
//     };
    
//     // Create the FITS file. The "!" prefix forces overwrite if the file exists.
//     if (fits_create_file(&fptr, ("!" + filename).c_str(), &status)) {
//         fits_report_error(stderr, status);
//         throw std::runtime_error("Error creating FITS file");
//     }
    
//     // Create a binary table extension named TELEMETRY.
//     if (fits_create_tbl(fptr, BINARY_TBL, nrows, ncols, ttype, tform_ptr, tunit, "TELEMETRY", &status)) {
//         fits_report_error(stderr, status);
//         throw std::runtime_error("Error creating telemetry table");
//     }
    
//     // --- Prepare data for each column ---
//     // COUNTER: a vector of integers (sample indices).
//     std::vector<int> counterCol(nrows);
//     for (long i = 0; i < nrows; ++i) {
//         counterCol[i] = i;
//     }
    
//     // TIMESTAMPS: convert the circular buffer to a vector.
//     std::vector<double> timestamps(telemetry.timestamp.begin(), telemetry.timestamp.end());
    
//     // LO_SERVO_MODE and HO_SERVO_MODE:
//     std::vector<int> loServo(telemetry.LO_servo_mode.begin(), telemetry.LO_servo_mode.end());
//     std::vector<int> hoServo(telemetry.HO_servo_mode.begin(), telemetry.HO_servo_mode.end());
    
//     // Helper lambda that flattens a circular buffer of Eigen::VectorXd.
//     auto flattenBuffer = [nrows](const boost::circular_buffer<Eigen::VectorXd>& buff, long vecSize) -> std::vector<double> {
//         std::vector<double> flat;
        
//         flat.reserve(nrows * vecSize);
//         for (const auto &v : buff) {
//             std::cout << "Sample vector size: " << v.size() << std::endl;
//             for (int j = 0; j < vecSize; j++) {
//                 flat.push_back(v(j));
//             }
//         }
//         return flat;
//     };
    
//     // Flatten each ring buffer into a 1D vector of doubles.
//     std::vector<double> imgCol    = flattenBuffer(telemetry.img, len_img);
//     std::vector<double> img_dmCol = flattenBuffer(telemetry.img_dm, len_img_dm);
//     std::vector<double> signalCol = flattenBuffer(telemetry.signal, len_signal);
//     std::vector<double> e_LOCol   = flattenBuffer(telemetry.e_LO, len_e_LO);
//     std::vector<double> u_LOCol   = flattenBuffer(telemetry.u_LO, len_u_LO);
//     std::vector<double> e_HOCol   = flattenBuffer(telemetry.e_HO, len_e_HO);
//     std::vector<double> u_HOCol   = flattenBuffer(telemetry.u_HO, len_u_HO);
//     std::vector<double> c_LOCol   = flattenBuffer(telemetry.c_LO, len_c_LO);
//     std::vector<double> c_HOCol   = flattenBuffer(telemetry.c_HO, len_c_HO);
    
//     // --- Write the columns to the FITS table ---
//     if (fits_write_col(fptr, TINT, 1, 1, 1, nrows, counterCol.data(), &status))
//         throw std::runtime_error("Error writing COUNTER column");
//     if (fits_write_col(fptr, TDOUBLE, 2, 1, 1, nrows, timestamps.data(), &status))
//         throw std::runtime_error("Error writing TIMESTAMPS column");
//     if (fits_write_col(fptr, TINT, 3, 1, 1, nrows, loServo.data(), &status))
//         throw std::runtime_error("Error writing LO_SERVO_MODE column");
//     if (fits_write_col(fptr, TINT, 4, 1, 1, nrows, hoServo.data(), &status))
//         throw std::runtime_error("Error writing HO_SERVO_MODE column");
//     if (fits_write_col(fptr, TDOUBLE, 5, 1, 1, nrows, imgCol.data(), &status))
//         throw std::runtime_error("Error writing IMG column");
//     if (fits_write_col(fptr, TDOUBLE, 6, 1, 1, nrows, img_dmCol.data(), &status))
//         throw std::runtime_error("Error writing IMG_DM column");
//     if (fits_write_col(fptr, TDOUBLE, 7, 1, 1, nrows, signalCol.data(), &status))
//         throw std::runtime_error("Error writing SIGNAL column");
//     if (fits_write_col(fptr, TDOUBLE, 8, 1, 1, nrows, e_LOCol.data(), &status))
//         throw std::runtime_error("Error writing E_LO column");
//     if (fits_write_col(fptr, TDOUBLE, 9, 1, 1, nrows, u_LOCol.data(), &status))
//         throw std::runtime_error("Error writing U_LO column");
//     if (fits_write_col(fptr, TDOUBLE, 10, 1, 1, nrows, e_HOCol.data(), &status))
//         throw std::runtime_error("Error writing E_HO column");
//     if (fits_write_col(fptr, TDOUBLE, 11, 1, 1, nrows, u_HOCol.data(), &status))
//         throw std::runtime_error("Error writing U_HO column");
//     if (fits_write_col(fptr, TDOUBLE, 12, 1, 1, nrows, c_LOCol.data(), &status))
//         throw std::runtime_error("Error writing C_LO column");
//     if (fits_write_col(fptr, TDOUBLE, 13, 1, 1, nrows, c_HOCol.data(), &status))
//         throw std::runtime_error("Error writing C_HO column");
    
//     // Write RMSE_EST
//     std::vector<double> rmse_est_vec(telemetry.rmse_est.begin(), telemetry.rmse_est.end());
//     if (fits_write_col(fptr, TDOUBLE, 14, 1, 1, nrows, rmse_est_vec.data(), &status))
//         throw std::runtime_error("Error writing RMSE_EST column");

//     // Write SNR
//     std::vector<double> snr_vec(telemetry.snr.begin(), telemetry.snr.end());
//     if (fits_write_col(fptr, TDOUBLE, 15, 1, 1, nrows, snr_vec.data(), &status))
//         throw std::runtime_error("Error writing SNR column");
        
//     // Close the FITS file.
//     if (fits_close_file(fptr, &status)) {
//         fits_report_error(stderr, status);
//         throw std::runtime_error("Error closing FITS file");
//     }
    
//     std::cout << "Telemetry successfully written to " << filename << std::endl;
//     return status;
// }


// // int write_telemetry_to_fits(const bdr_telem &telemetry, const std::string &filename) {
// //     fitsfile *fptr = nullptr;
// //     int status = 0;
    
// //     // Determine the number of telemetry samples (rows) from the timestamp ring buffer.
// //     long nrows = telemetry.timestamp.size();
// //     if (nrows == 0) {
// //         std::cerr << "No telemetry data available to write." << std::endl;
// //         return 0;
// //     }
    
// //     // Helper lambda to extract the length of an Eigen vector from a circular buffer.
// //     auto getVecLength = [](const boost::circular_buffer<Eigen::VectorXd> &buff) -> long {
// //         return (buff.empty() ? 0 : buff.front().size());
// //     };

// //     long len_img     = getVecLength(telemetry.img);
// //     long len_img_dm  = getVecLength(telemetry.img_dm);
// //     long len_signal  = getVecLength(telemetry.signal);
// //     long len_e_LO    = getVecLength(telemetry.e_LO);
// //     long len_u_LO    = getVecLength(telemetry.u_LO);
// //     long len_e_HO    = getVecLength(telemetry.e_HO);
// //     long len_u_HO    = getVecLength(telemetry.u_HO);
// //     long len_c_LO    = getVecLength(telemetry.c_LO);
// //     long len_c_HO    = getVecLength(telemetry.c_HO);
    
// //     // Lambda to create a TFORM string for an array of doubles of fixed length.
// //     auto makeFormat = [=](long len) -> std::string {
// //         return std::to_string(len) + "D"; // For example, "1024D"
// //     };

// //     std::string form_img     = makeFormat(len_img);
// //     std::string form_img_dm  = makeFormat(len_img_dm);
// //     std::string form_signal  = makeFormat(len_signal);
// //     std::string form_e_LO    = makeFormat(len_e_LO);
// //     std::string form_u_LO    = makeFormat(len_u_LO);
// //     std::string form_e_HO    = makeFormat(len_e_HO);
// //     std::string form_u_HO    = makeFormat(len_u_HO);
// //     std::string form_c_LO    = makeFormat(len_c_LO);
// //     std::string form_c_HO    = makeFormat(len_c_HO);
    
// //     // Number of columns in the FITS table.
// //     const int ncols = 13;
    
// //     // Define column names (TTYPE keywords).
// //     char *ttype[ncols] = {
// //         const_cast<char*>("COUNTER"),
// //         const_cast<char*>("TIMESTAMPS"),
// //         const_cast<char*>("LO_SERVO_MODE"),
// //         const_cast<char*>("HO_SERVO_MODE"),
// //         const_cast<char*>("IMG"),
// //         const_cast<char*>("IMG_DM"),
// //         const_cast<char*>("SIGNAL"),
// //         const_cast<char*>("E_LO"),
// //         const_cast<char*>("U_LO"),
// //         const_cast<char*>("E_HO"),
// //         const_cast<char*>("U_HO"),
// //         const_cast<char*>("C_LO"),
// //         const_cast<char*>("C_HO")
// //     };
    
// //     // Create a two-dimensional array for the TFORM keywords.
// //     char tform[ncols][20];
// //     snprintf(tform[0], sizeof(tform[0]), "1J");       // COUNTER: integer
// //     snprintf(tform[1], sizeof(tform[1]), "1D");       // TIMESTAMPS: double
// //     snprintf(tform[2], sizeof(tform[2]), "1J");       // LO_SERVO_MODE: integer
// //     snprintf(tform[3], sizeof(tform[3]), "1J");       // HO_SERVO_MODE: integer
// //     snprintf(tform[4], sizeof(tform[4]), "%s", form_img.c_str());
// //     snprintf(tform[5], sizeof(tform[5]), "%s", form_img_dm.c_str());
// //     snprintf(tform[6], sizeof(tform[6]), "%s", form_signal.c_str());
// //     snprintf(tform[7], sizeof(tform[7]), "%s", form_e_LO.c_str());
// //     snprintf(tform[8], sizeof(tform[8]), "%s", form_u_LO.c_str());
// //     snprintf(tform[9], sizeof(tform[9]), "%s", form_e_HO.c_str());
// //     snprintf(tform[10], sizeof(tform[10]), "%s", form_u_HO.c_str());
// //     snprintf(tform[11], sizeof(tform[11]), "%s", form_c_LO.c_str());
// //     snprintf(tform[12], sizeof(tform[12]), "%s", form_c_HO.c_str());
    
// //     // Create an array of char* pointers pointing to each row of tform.
// //     char* tform_ptr[ncols];
// //     for (int i = 0; i < ncols; ++i) {
// //         tform_ptr[i] = tform[i];
// //     }
    
// //     // Column units (optional; set to empty if not used).
// //     char *tunit[ncols] = {
// //         const_cast<char*>(""),
// //         const_cast<char*>("microsec"),
// //         const_cast<char*>(""),
// //         const_cast<char*>(""),
// //         const_cast<char*>(""),
// //         const_cast<char*>(""),
// //         const_cast<char*>(""),
// //         const_cast<char*>(""),
// //         const_cast<char*>(""),
// //         const_cast<char*>(""),
// //         const_cast<char*>(""),
// //         const_cast<char*>(""),
// //         const_cast<char*>("")
// //     };
    
// //     // Create (or overwrite) the FITS file.
// //     if (fits_create_file(&fptr, ("!" + filename).c_str(), &status)) {
// //         fits_report_error(stderr, status);
// //         throw std::runtime_error("Error creating FITS file");
// //     }
    
// //     // Create a binary table extension named TELEMETRY.
// //     if (fits_create_tbl(fptr, BINARY_TBL, nrows, ncols, ttype, tform_ptr, tunit, "TELEMETRY", &status)) {
// //         fits_report_error(stderr, status);
// //         throw std::runtime_error("Error creating telemetry table");
// //     }
    
// //     // Prepare data for each column.
// //     // COUNTER: Use sample indices.
// //     std::vector<int> counterCol(nrows);
// //     for (long i = 0; i < nrows; ++i) {
// //         counterCol[i] = i;
// //     }
    
// //     // TIMESTAMPS: Convert circular buffer to vector.
// //     std::vector<double> timestamps(telemetry.timestamp.begin(), telemetry.timestamp.end());
    
// //     // LO_SERVO_MODE and HO_SERVO_MODE:
// //     std::vector<int> loServo(telemetry.LO_servo_mode.begin(), telemetry.LO_servo_mode.end());
// //     std::vector<int> hoServo(telemetry.HO_servo_mode.begin(), telemetry.HO_servo_mode.end());
    
// //     // Helper lambda to flatten a circular buffer of Eigen::VectorXd.
// //     auto flattenBuffer = [nrows](const boost::circular_buffer<Eigen::VectorXd>& buff, long vecSize) -> std::vector<double> {
// //         std::vector<double> flat;
// //         flat.reserve(nrows * vecSize);
// //         for (const auto &v : buff) {
// //             for (int j = 0; j < vecSize; j++) {
// //                 flat.push_back(v(j));
// //             }
// //         }
// //         return flat;
// //     };
    
// //     std::vector<double> imgCol    = flattenBuffer(telemetry.img, len_img);
// //     std::vector<double> img_dmCol = flattenBuffer(telemetry.img_dm, len_img_dm);
// //     std::vector<double> signalCol = flattenBuffer(telemetry.signal, len_signal);
// //     std::vector<double> e_LOCol   = flattenBuffer(telemetry.e_LO, len_e_LO);
// //     std::vector<double> u_LOCol   = flattenBuffer(telemetry.u_LO, len_u_LO);
// //     std::vector<double> e_HOCol   = flattenBuffer(telemetry.e_HO, len_e_HO);
// //     std::vector<double> u_HOCol   = flattenBuffer(telemetry.u_HO, len_u_HO);
// //     std::vector<double> c_LOCol   = flattenBuffer(telemetry.c_LO, len_c_LO);
// //     std::vector<double> c_HOCol   = flattenBuffer(telemetry.c_HO, len_c_HO);
    
// //     // Write each column to the table.
// //     if (fits_write_col(fptr, TINT, 1, 1, 1, nrows, counterCol.data(), &status))
// //         throw std::runtime_error("Error writing COUNTER column");
// //     if (fits_write_col(fptr, TDOUBLE, 2, 1, 1, nrows, timestamps.data(), &status))
// //         throw std::runtime_error("Error writing TIMESTAMPS column");
// //     if (fits_write_col(fptr, TINT, 3, 1, 1, nrows, loServo.data(), &status))
// //         throw std::runtime_error("Error writing LO_SERVO_MODE column");
// //     if (fits_write_col(fptr, TINT, 4, 1, 1, nrows, hoServo.data(), &status))
// //         throw std::runtime_error("Error writing HO_SERVO_MODE column");
// //     if (fits_write_col(fptr, TDOUBLE, 5, 1, 1, nrows, imgCol.data(), &status))
// //         throw std::runtime_error("Error writing IMG column");
// //     if (fits_write_col(fptr, TDOUBLE, 6, 1, 1, nrows, img_dmCol.data(), &status))
// //         throw std::runtime_error("Error writing IMG_DM column");
// //     if (fits_write_col(fptr, TDOUBLE, 7, 1, 1, nrows, signalCol.data(), &status))
// //         throw std::runtime_error("Error writing SIGNAL column");
// //     if (fits_write_col(fptr, TDOUBLE, 8, 1, 1, nrows, e_LOCol.data(), &status))
// //         throw std::runtime_error("Error writing E_LO column");
// //     if (fits_write_col(fptr, TDOUBLE, 9, 1, 1, nrows, u_LOCol.data(), &status))
// //         throw std::runtime_error("Error writing U_LO column");
// //     if (fits_write_col(fptr, TDOUBLE, 10, 1, 1, nrows, e_HOCol.data(), &status))
// //         throw std::runtime_error("Error writing E_HO column");
// //     if (fits_write_col(fptr, TDOUBLE, 11, 1, 1, nrows, u_HOCol.data(), &status))
// //         throw std::runtime_error("Error writing U_HO column");
// //     if (fits_write_col(fptr, TDOUBLE, 12, 1, 1, nrows, c_LOCol.data(), &status))
// //         throw std::runtime_error("Error writing C_LO column");
// //     if (fits_write_col(fptr, TDOUBLE, 13, 1, 1, nrows, c_HOCol.data(), &status))
// //         throw std::runtime_error("Error writing C_HO column");
    
// //     // Close the FITS file.
// //     if (fits_close_file(fptr, &status)) {
// //         fits_report_error(stderr, status);
// //         throw std::runtime_error("Error closing FITS file");
// //     }
    
// //     std::cout << "Telemetry successfully written to " << filename << std::endl;
// //     return status;
// // }


// // The main RTC function

// void telemetry(){
//     // Telemetry thread loop.
//     while (servo_mode != SERVO_STOP) {
//         // Sleep for a fixed interval (e.g., 1 second) between telemetry writes.
//         std::this_thread::sleep_for(std::chrono::seconds(1));

//         if (rtc_config.state.take_telemetry==1) {
//             // Get current system time for file naming.
//             auto now = std::chrono::system_clock::now();                
//             std::time_t t = std::chrono::system_clock::to_time_t(now);
//             std::tm tm = *std::localtime(&t);
//             std::ostringstream oss;
//             if (telemFormat == "fits") {
//                 oss << "/home/asg/Music/telemetry_" 
//                     << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".fits";
//             } else { // default to "json"
//                 oss << "/home/asg/Music/telemetry_" 
//                     << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".json";
//             }
//             std::string filename = oss.str();

//             bdr_telem currentTelem;
//             {   // Lock telemetry mutex and copy current telemetry.
//                 std::lock_guard<std::mutex> lock(telemetry_mutex);
//                 currentTelem = rtc_config.telem;
//             }
            
//             if (telemFormat == "fits") {
//                 // Call your FITS writer.
//                 try {
//                     int status = write_telemetry_to_fits(currentTelem, filename);
//                     std::cout << "Telemetry written to FITS file: " << filename << "with output" << status << std::endl;
//                 } catch (const std::exception &ex) {
//                     std::cerr << "Error writing telemetry FITS file: " << ex.what() << std::endl;
//                 }
//             } else {
//                 // Convert telemetry to JSON and write it.
//                 json j = telemetry_to_json(currentTelem);
//                 std::ofstream ofs(filename);
//                 if (ofs.is_open()) {
//                     ofs << j.dump(4);
//                     ofs.close();
//                     std::cout << "Telemetry written to " << filename << std::endl;
//                 } else {
//                     std::cerr << "Error opening file " << filename << " for writing." << std::endl;
//                 }
//             }
//             // Reset the take_telemetry flag.
//             rtc_config.state.take_telemetry = 0;
//         }
//     }
// }

// // void telemetry(){
// //     // Temporary variabiles that don't need initialisation
// //     // can go here.
// //     //initialise_telemetry();

// //     while (servo_mode != SERVO_STOP) {  // keep_going is a global flag indicating the system is running


// //         // Sleep for a fixed interval (say, 1 second) before processing telemetry
// //         std::this_thread::sleep_for(std::chrono::seconds(1));

// //         const std::string filename = "telemetry.json";
        
// //         if (rtc_config.state.take_telemetry){

// //             auto now = std::chrono::system_clock::now();                
// //             std::time_t t = std::chrono::system_clock::to_time_t(now);
// //             std::tm tm = *std::localtime(&t);

// //             std::ostringstream oss;
// //             oss << "/home/asg/Music/telemetry_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".json";
// //             std::string filename = oss.str();

// //             bdr_telem currentTelem;
// //             {
// //                 // Lock telemetry mutex, copy the telemetry object.
// //                 std::lock_guard<std::mutex> lock(telemetry_mutex);
// //                 currentTelem = rtc_config.telem;  // Ensure your telemetry struct supports copy.
// //             }
            
// //             // Convert the telemetry data to JSON.
// //             json j = telemetry_to_json(currentTelem);
            
// //             // Write to file.
// //             std::ofstream ofs(filename);
// //             if (ofs.is_open()) {
// //                 ofs << j.dump(4);
// //                 ofs.close();
// //                 std::cout << "Telemetry written to " << filename << std::endl;
// //             } else {
// //                 std::cerr << "Error opening " << filename << " for writing." << std::endl;
// //             }
// //             std::cout << "Done writing " << std::endl;
// //             // set back to false
// //             rtc_config.state.take_telemetry = 0;
// //         }
// //     }
// // }
