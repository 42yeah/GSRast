echo "$args.hpp"
echo "#pragma once" | Out-File -encoding UTF8 "$args.hpp"
echo "#include ""$args.hpp""" | Out-File -encoding UTF8 "$args.cpp"
