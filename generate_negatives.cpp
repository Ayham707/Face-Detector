#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <windows.h> // for FindFirstFile/FindNextFile

bool ends_with(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main() {
    std::string folder = "C:/Users/ayham/Face-Detector/dataset/negatives";
    std::ofstream out("negatives.txt");
    if (!out.is_open()) {
        std::cerr << "Cannot open negatives.txt!\n";
        return -1;
    }

    WIN32_FIND_DATA fileData;
    HANDLE hFind = FindFirstFile((folder + "/*.*").c_str(), &fileData);
    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "Folder not found!\n";
        return -1;
    }

    std::vector<std::string> files;
    do {
        std::string name = fileData.cFileName;
        if (name != "." && name != "..") {
            std::string lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

            if (ends_with(lower, ".jpg") || ends_with(lower, ".png") || ends_with(lower, ".jpeg")) {
                files.push_back(folder + "/" + name);
            }
        }
    } while (FindNextFile(hFind, &fileData));
    FindClose(hFind);

    std::sort(files.begin(), files.end());
    for (auto &f : files) out << f << "\n";

    out.close();
    std::cout << "negatives.txt successfully created!\n";
    return 0;
}
