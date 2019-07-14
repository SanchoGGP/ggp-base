package org.ggp.base.player.gamer.statemachine.sancho.pool;

/**
 * An pool of items.
 *
 * @param <ItemType> - the type of items stored in this pool.
 */
public interface Pool<ItemType>
{
  /**
   * Interface to be implemented by classes capable of allocating (and resetting) objects in a pool.
   *
   * @param <ItemType> the type of item to be allocated.
   */
  public interface ObjectAllocator<ItemType>
  {
    /**
     * @return a newly allocated object.
     *
     * @param xiPoolIndex - index in the pool from which this object was allocated.
     */
    public ItemType newObject(int xiPoolIndex);

    /**
     * Reset an object, ready for re-use.
     *
     * @param xiObject - the object to reset.
     * @param xiFree   - whether to free the internals.
     */
    public void resetObject(ItemType xiObject, boolean xiFree);

    /**
     *
     * @param xiObject - the object
     * @return whether an object should be reset.
     */
    public boolean shouldReset(ItemType xiObject);
  }

  /**
   * Allocate a new item from the pool.
   *
   * @param xiAllocator - object allocator to use if no new items are available.
   *
   * @return the new item.
   */
  public ItemType allocate(ObjectAllocator<ItemType> xiAllocator);

  /**
   * Return an item to the pool.
   *
   * The pool promises to call resetObject() for any freed items before re-use.
   *
   * @param xiItem  - the item.
   * @param xiIndex - index of the item being freed.
   */
  public void free(ItemType xiItem, int xiIndex);

 /**
   * Clear the pool - resetting all items that are still allocated.
   *
   * @param xiAllocator - an object allocator.
   * @param xiFilter    - whether to filter the items to be reset (used to reset nodes from just one MCTSTree).
   */
  public void clear(ObjectAllocator<ItemType> xiAllocator, boolean xiFilter);

  /**
   * Optional method to get the pool capacity.
   *
   * @return the capacity of the pool.
   */
  public int getCapacity();

  /**
   * @return whether the pool is (nearly) full.
   *
   * When full, the caller needs to free() some items to ensure that subsequently allocations will continue to succeed.
   */
  public boolean isFull();

  /**
   * Set a minimum free node requirement to report non-full.
   *
   * @param xiThreshold - the threshold.
   */
  public void setNonFreeThreshold(int xiThreshold);

  /**
   * @return the number of items currently in use.
   */
  public int getNumItemsInUse();

  /**
   * @return the percentage of this pool that is in use.
   */
  public int getPoolUsage();

  /**
   * Optional method to retrieve by index an item that has already been allocated.
   *
   * @param xiIndex - the index
   *
   * @return the item.
   *
   */
  public ItemType get(int xiNodeRef);
}
