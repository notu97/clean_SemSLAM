/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Symbol.cpp
 * @date June 9, 2012
 * @author: Frank Dellaert
 * @author: Richard Roberts
 */

#include "semantic_slam/Symbol.h"

#include <boost/format.hpp>

#include <iostream>
#include <limits.h>

static const size_t keyBits = sizeof(Key) * 8;
static const size_t chrBits = sizeof(unsigned char) * 8;
static const size_t indexBits = keyBits - chrBits;
static const Key chrMask =
  Key(UCHAR_MAX) << indexBits; // For some reason, std::numeric_limits<unsigned
                               // char>::max() fails
static const Key indexMask = ~chrMask;

Symbol::Symbol(Key key)
  : c_((unsigned char)((key & chrMask) >> indexBits))
  , j_(key & indexMask)
{}

Key
Symbol::key() const
{
    if (j_ > indexMask) {
        boost::format msg("Symbol index is too large, j=%d, indexMask=%d");
        msg % j_ % indexMask;
        throw std::invalid_argument(msg.str());
    }
    Key key = (Key(c_) << indexBits) | j_;
    return key;
}

void
Symbol::print(const std::string& s) const
{
    std::cout << s << (std::string)(*this) << std::endl;
}

bool
Symbol::equals(const Symbol& expected, double tol) const
{
    return (*this) == expected;
}

Symbol::operator std::string() const
{
    return str(boost::format("%c%d") % c_ % j_);
}

// static Symbol make(gtsam::Key key) { return Symbol(key);}

std::function<bool(Key)>
Symbol::ChrTest(unsigned char c)
{
    return [c](unsigned char other) { return Symbol(other).chr() == c; };
}
