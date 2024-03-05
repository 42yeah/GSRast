#include "Database.hpp"
#include <cstdio>

#define PRINT_LMDB_ERR(test) ((test) ? (void) 0 : (void) fprintf(stderr, "LMDB error: %s\n", mdb_strerror(_rc)))
#define CLE(expr) PRINT_LMDB_ERR((_rc = (expr)) == MDB_SUCCESS)

Database::Ptr Database::_database = nullptr;

#include "Database.hpp"
#include "lmdb.h"
#include <cstring>

Database::Database(const std::string &path) : _env(nullptr), _cur(nullptr)
{
    CLE(mdb_env_create(&_env));
    CLE(mdb_env_set_maxreaders(_env, 8));
    CLE(mdb_env_set_mapsize(_env, 10485760));
    CLE(mdb_env_set_maxdbs(_env, 1024));
    CLE(mdb_env_open(_env, path.c_str(), MDB_FIXEDMAP, 0664));
}

Database::~Database()
{
    // mdb_txn_abort(_txn);
    // mdb_dbi_close(_env, _dbi);
    mdb_env_close(_env);
}

Database::Ptr Database::get()
{
    if (_database == nullptr)
    {
        _database = std::make_shared<Database>(DEFAULT_DB_PATH);
    }
    return _database;
}

void Database::put(const std::string &db, const std::string &key, const char *value, size_t valSize)
{
    MDB_txn *txn;
    MDB_dbi dbi;
    CLE(mdb_txn_begin(_env, nullptr, 0, &txn));
    CLE(mdb_dbi_open(txn, db == "" ? nullptr : db.c_str(), MDB_CREATE, &dbi));
    _key.mv_size = key.size();
    _key.mv_data = const_cast<char *>(key.c_str());
    _data.mv_size = valSize;
    _data.mv_data = const_cast<char *>(value);
    CLE(mdb_put(txn, dbi, &_key, &_data, 0));
    CLE(mdb_txn_commit(txn));
    mdb_dbi_close(_env, dbi);
}

int Database::get(const std::string &db, const std::string &key, char *value, size_t valSize)
{
    if (fetch(db, key) != MDB_SUCCESS)
    {
        return _rc;
    }
    memset(value, 0, valSize);
    size_t copySize = valSize > _data.mv_size ? _data.mv_size : valSize;
    memcpy(value, _data.mv_data, copySize);
    return _rc;
}

MDB_val Database::getRaw(const std::string &db, const std::string &key)
{
    if (fetch(db, key) != MDB_SUCCESS)
    {
        _data.mv_size = 0;
        _data.mv_data = nullptr;
    }
    return _data;
}

int Database::fetch(const std::string &db, const std::string &key)
{
    MDB_txn *txn;
    MDB_dbi dbi;
    CLE(mdb_txn_begin(_env, nullptr, MDB_RDONLY, &txn));
    CLE(mdb_dbi_open(txn, db == "" ? nullptr : db.c_str(), MDB_CREATE, &dbi));
    if (_rc != MDB_SUCCESS)
    {
        mdb_txn_abort(txn);
        return _rc;
    }
    _key.mv_size = key.size();
    _key.mv_data = const_cast<char *>(key.c_str());
    _rc = mdb_get(txn, dbi, &_key, &_data);
    mdb_txn_abort(txn);
    mdb_dbi_close(_env, dbi);
    PRINT_LMDB_ERR(_rc == MDB_SUCCESS || _rc == MDB_NOTFOUND);
    return _rc;
}

std::string Database::getString(const std::string &db, const std::string &key)
{
    if ((_rc = fetch(db, key)) != MDB_SUCCESS && _rc != MDB_NOTFOUND)
    {
        return "Database error";
    }
    if (_rc == MDB_NOTFOUND)
    {
        return "";
    }
    std::string str;
    str.append(reinterpret_cast<char *>(_data.mv_data), _data.mv_size);
    return str;
}

int Database::remove(const std::string &db, const std::string &key)
{
    MDB_txn *txn;
    MDB_dbi dbi;
    CLE(mdb_txn_begin(_env, nullptr, 0, &txn));
    CLE(mdb_dbi_open(txn, db == "" ? nullptr : db.c_str(), MDB_CREATE, &dbi));
    _key.mv_size = key.size();
    _key.mv_data = const_cast<char *>(key.c_str());
    CLE(mdb_del(txn, dbi, &_key, nullptr));
    if (_rc != MDB_SUCCESS)
    {
        mdb_txn_abort(txn);
        mdb_dbi_close(_env, dbi);
        return _rc;
    }
    CLE(mdb_txn_commit(txn));
    mdb_dbi_close(_env, dbi);
    return _rc;
}

int Database::drop(const std::string &db)
{
    MDB_txn *txn;
    MDB_dbi dbi;
    CLE(mdb_txn_begin(_env, nullptr, 0, &txn));
    CLE(mdb_dbi_open(txn, db == "" ? nullptr : db.c_str(), MDB_CREATE, &dbi));
    if (_rc != MDB_SUCCESS)
    {
        mdb_txn_abort(txn);
        return _rc;
    }
    CLE(mdb_drop(txn, dbi, 1));
    if (_rc != MDB_SUCCESS)
    {
        mdb_txn_abort(txn);
        return _rc;
    }
    CLE(mdb_txn_commit(txn));
    return _rc;
}

void Database::iterate(const std::string &db, std::function<void(std::string, MDB_val)> iterator, bool includeHidden)
{
    MDB_txn *txn;
    MDB_dbi dbi;
    CLE(mdb_txn_begin(_env, nullptr, MDB_RDONLY, &txn));
    CLE(mdb_dbi_open(txn, db == "" ? nullptr : db.c_str(), 0, &dbi));
    if (_rc != MDB_SUCCESS)
    {
        fprintf(stderr, "Cannot iterate: cannot open %s\n", db.c_str());
        mdb_txn_abort(txn);
        return;
    }
    CLE(mdb_cursor_open(txn, dbi, &_cur));

    while ((_rc = mdb_cursor_get(_cur, &_key, &_data, MDB_NEXT)) == 0)
    {
        char *keyChar = reinterpret_cast<char *>(_key.mv_data);
        if (!includeHidden && _key.mv_size >= 2 && keyChar[0] == '_' && keyChar[1] == '_')
        {
            continue;
        }
        std::string k;
        k.append(reinterpret_cast<char *>(_key.mv_data), _key.mv_size);
        iterator(k, _data);
    }
    PRINT_LMDB_ERR(_rc == MDB_NOTFOUND);
    mdb_cursor_close(_cur);
    mdb_txn_abort(txn);
    mdb_dbi_close(_env, dbi);
}
