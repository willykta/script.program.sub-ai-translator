import urllib3
import xbmc
from .config import CONNECTION_POOL_CONFIG

class ConnectionPoolManager:
    def __init__(self, config=None):
        self.config = config or CONNECTION_POOL_CONFIG
        self.pools = {}
        self.default_pool = self._create_pool()
        
    def _create_pool(self, num_pools=None, maxsize=None, retries=False):
        """Create a connection pool with specified settings"""
        # Use provided values or fall back to config
        num_pools = num_pools or self.config["num_pools"]
        maxsize = maxsize or self.config["maxsize"]
        
        try:
            pool = urllib3.PoolManager(
                num_pools=num_pools,
                maxsize=maxsize,
                retries=retries,  # We handle retries ourselves
                timeout=urllib3.Timeout(
                    connect=self.config["connect_timeout"],
                    read=self.config["read_timeout"]
                ),
                block=False
            )
            xbmc.log(f"[CONNECTION_POOL] Created pool with num_pools={num_pools}, maxsize={maxsize}", xbmc.LOGDEBUG)
            return pool
        except Exception as e:
            xbmc.log(f"[CONNECTION_POOL] Failed to create pool: {str(e)}", xbmc.LOGERROR)
            # Fallback to None
            return None
    
    def get_pool(self, host=None):
        """Get a connection pool for a specific host"""
        if self.default_pool is None:
            return None
            
        if host is None:
            return self.default_pool
            
        if host not in self.pools:
            try:
                # For different hosts, we might want to create specific pools
                # For now, we'll use the default pool for all hosts
                self.pools[host] = self.default_pool
                xbmc.log(f"[CONNECTION_POOL] Using default pool for host: {host}", xbmc.LOGDEBUG)
            except Exception as e:
                xbmc.log(f"[CONNECTION_POOL] Failed to create pool for {host}: {str(e)}", xbmc.LOGERROR)
                return self.default_pool
                
        return self.pools[host]

# Global instance
connection_pool_manager = ConnectionPoolManager()