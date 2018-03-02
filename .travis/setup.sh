set -ex

export top_dir=$(dirname ${0%/*})

source "${HOME}/virtualenv/bin/activate"
python --version

# setup ccache
if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  export PATH="/usr/lib/ccache:$PATH"
elif [ "$TRAVIS_OS_NAME" == "osx" ]; then
  export PATH="/usr/local/opt/ccache/libexec:$PATH"
else
  echo Unknown OS: $TRAVIS_OS_NAME
  exit 1
fi
ccache --max-size 1G
