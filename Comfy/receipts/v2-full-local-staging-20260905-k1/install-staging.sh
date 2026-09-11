set -eu
actual="$(wp --skip-plugins --skip-themes option get siteurl)"
if [ "$actual" != "https://staging-7e48-skyyrose.wpcomstaging.com" ]; then
  printf '%s\n' "STOP: remote WordPress identity does not match authorized staging." >&2
  exit 80
fi
active="$(wp --skip-plugins --skip-themes option get stylesheet)"
if [ "$active" != "skyyrose-flagship-2" ]; then
  printf '%s\n' "STOP: active staging theme changed." >&2
  exit 81
fi
php -r '$base="/srv/htdocs/wp-content/themes";
$name="skyyrose-flagship-2";
$live="$base/$name";
$id="v2-full-local-staging-20260905-k1";
$incoming="$base/.$id.incoming";
$backup="$base/.$id.previous";
$archive="/home/152046411/$id.tar";
$expected=json_decode(base64_decode("eyJmdW5jdGlvbnMucGhwIjoiYWMyYmEyMWZjY2ZjYzY1OGRiZDZmMzA5YjBiYzY3MWI5ZmFiOGJkNTdlMmMwYTFiMzYyMWQ2MGE0YmJmZDI1NiIsInRlbXBsYXRlLWNvbGxlY3Rpb24ucGhwIjoiZWQ4ZTllOTU4NjEwZTFmN2M4ZjFkZWE1YjJkYWQxNzNkNWVhOWM0MzE1ZDcwODI2NzljMTJmYjk1NTNkNDNhZCIsInRlbXBsYXRlLXBhcnRzL2ltbWVyc2l2ZS93b3JsZC5waHAiOiJhMWYwOGQ5MTY4NTE1ODY4NzhjMGRkYmNjOTZhZmNlNDkwZTdlYmIyZjJjZGI0YTA1ZDY1OTY1NDZkOGZmZTg4In0="),true);
$lock=fopen("/home/152046411/.$id.lock","c");
if(!$lock||!flock($lock,LOCK_EX|LOCK_NB)){fwrite(STDERR,"Another staging release owns the lock.\n");exit(71);}
foreach($expected as $file=>$hash){
 if(!is_file("$live/$file")||!hash_equals($hash,hash_file("sha256","$live/$file"))){
  fwrite(STDERR,"STOP: staging changed since deployment preparation: $file\n");exit(72);
 }
}
if(is_dir($incoming)||is_dir($backup)){fwrite(STDERR,"Existing release directories require reconciliation.\n");exit(73);}
if(!mkdir($incoming,0700)){exit(74);}
$command="tar -xf ".escapeshellarg($archive)." -C ".escapeshellarg($incoming);
passthru($command,$rc);if($rc!==0){exit($rc);}
$next="$incoming/$name";
foreach(["style.css","functions.php","data/collection-scene-motion.json","inc/hero-commerce-scenes.php","assets/js/collection-scene-motion.min.js"] as $required){
 if(!is_file("$next/$required")){fwrite(STDERR,"Incomplete release payload: $required\n");exit(75);}
}
$entries=new RecursiveIteratorIterator(new RecursiveDirectoryIterator($next,FilesystemIterator::SKIP_DOTS),RecursiveIteratorIterator::SELF_FIRST);
foreach($entries as $entry){
 if($entry->isLink()){fwrite(STDERR,"Symbolic links are not allowed in the release.\n");exit(76);}
 if($entry->isDir()){chmod($entry->getPathname(),0755);}else{chmod($entry->getPathname(),0644);}
}
chmod($next,0755);
if(!rename($live,$backup)){fwrite(STDERR,"Could not preserve staging rollback directory.\n");exit(77);}
if(!rename($next,$live)){rename($backup,$live);fwrite(STDERR,"Activation failed; previous staging theme restored.\n");exit(78);}
$privateBackup="/home/152046411/$id.previous";
$backupPath=$backup;
if(rename($backup,$privateBackup)){$backupPath=$privateBackup;}else{chmod($backup,0700);}
rmdir($incoming);
echo json_encode(["status"=>"STAGING_THEME_ACTIVATED","theme"=>$live,"rollback_directory"=>$backupPath,"release"=>$id])."\n";
'
wp --skip-plugins --skip-themes cache flush
wp --skip-themes eval '$ids=[9822,10412,10413,10414,10416,10417,10418,9454,9455,9456,10,11,12];foreach($ids as $id){clean_post_cache($id);}do_action("litespeed_purge_all");echo "Staging page-cache invalidation requested.\n";'
