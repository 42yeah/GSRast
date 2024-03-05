#pragma once

#include "Config.hpp"
#include <lmdb.h>
#include <functional>

#define DEFAULT_DB_PATH "."

/**
 * I take care of key-based database storage.
 * I have all the regular functionalities of a full-fledged DB;
 * CRUD, and the ability to add/drop databases.
 * I am a global singleton (bad idea, I know) but hey, I don't
 * really think we need multiple here.
 */
class Database
{
public:
    CLASS_PTRS(Database)

    Database(const std::string &path);
    ~Database();

    void put(const std::string &db, const std::string &key, const char *value, size_t valSize);
    int get(const std::string &db, const std::string &key, char *value, size_t valSize);
    int remove(const std::string &db, const std::string &key);
    MDB_val getRaw(const std::string &db, const std::string &key);
    std::string getString(const std::string &db, const std::string &key);
    void iterate(const std::string &db, std::function<void(std::string, MDB_val)> iterator, bool includeHidden = false);
    int drop(const std::string &db);

    static Database::Ptr get();

protected:
    /**
     * I fetch data into our internal shared variable, but nothing will be done yet.
     */
    int fetch(const std::string &db, const std::string &key);

private:
    MDB_env *_env;
    MDB_val _key, _data;
    MDB_stat _mst;
    MDB_cursor *_cur;
    int _rc;

    static Database::Ptr _database;
};
